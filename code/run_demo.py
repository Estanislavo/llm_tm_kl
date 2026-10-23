import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from aartm_q.preprocessing import DatasetPreprocessor
from aartm_q.semantic_aartm import fit_semantic_aartm, word_topic_probabilities


def main():
    parser = argparse.ArgumentParser(description='Run the original small-corpus AARTM+Q update on a synthetic text collection')
    parser.add_argument('--documents', type=Path, default=Path(__file__).resolve().parents[1] / 'data/demo_documents.json')
    parser.add_argument('--topics', type=int, default=3)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    documents = json.loads(args.documents.read_text(encoding='utf-8'))
    prep = DatasetPreprocessor()
    data, bounds = prep.fit_transform(documents)
    vocabulary = prep.vocabulary
    chunks = []
    for doc in documents:
        words = doc.split()
        chunks.extend(' '.join(words[i:i+100]) for i in range(0, len(words), 100))
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
    embeddings = encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    centers = KMeans(n_clusters=args.topics, random_state=args.seed, n_init=10).fit(embeddings).cluster_centers_
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-30)
    assignments = softmax(embeddings @ centers.T / 0.05, axis=1)
    counts = np.zeros((len(chunks), len(vocabulary)))
    for i, tokens in enumerate(prep.tokenize(chunks)):
        for word, count in Counter(tokens).items():
            if word in vocabulary:
                counts[i, vocabulary[word]] = count
    q = counts.T @ assignments + 1e-8
    q = (q / q.sum(axis=1, keepdims=True)).astype(np.float32)
    phi, n_t = fit_semantic_aartm(data, bounds, q, seed=args.seed, initialization=0.95, kl_strength=0.1, steps=args.steps)
    phi = np.asarray(phi)
    assert np.isfinite(phi).all() and (phi >= 0).all()
    assert np.allclose(phi.sum(axis=1), 1, atol=1e-5)
    beta = word_topic_probabilities(phi, data)
    words = sorted(vocabulary, key=vocabulary.get)
    topics = [[words[i] for i in np.argsort(beta[:, t])[-10:][::-1]] for t in range(args.topics)]
    result = dict(purpose='synthetic execution example; excluded from paper results', documents=len(documents), vocabulary=len(words), steps=args.steps, seed=args.seed, topics=topics)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, phi=phi, q=q, n_t=np.asarray(n_t), words=np.array(words))


if __name__ == '__main__':
    main()

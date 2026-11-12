import argparse
from collections import Counter
import json
from pathlib import Path
import hashlib

import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from aartm_q.preprocessing import DatasetPreprocessor
from aartm_q.semantic_aartm import fit_semantic_aartm, word_topic_probabilities


def completion_perplexity(beta, tokenized, vocabulary):
    beta = 0.9999 * beta + 0.0001 / len(vocabulary)
    log_probability, held_out = 0.0, 0
    for tokens in tokenized:
        observed = Counter(vocabulary[w] for i, w in enumerate(tokens) if i % 5 and w in vocabulary)
        hidden = Counter(vocabulary[w] for i, w in enumerate(tokens) if i % 5 == 0 and w in vocabulary)
        theta = np.full(beta.shape[1], 1 / beta.shape[1])
        for _ in range(40):
            counts = np.zeros(beta.shape[1])
            for word, count in observed.items():
                posterior = beta[word] * theta
                counts += count * posterior / posterior.sum()
            theta = (counts + 0.1) / (counts.sum() + 0.1 * len(theta))
        for word, count in hidden.items():
            log_probability += count * np.log(max(beta[word] @ theta, 1e-30))
            held_out += count
    if not held_out:
        raise ValueError('No held-out tokens in the training vocabulary')
    return float(np.exp(-log_probability / held_out)), held_out


def main():
    parser = argparse.ArgumentParser(description='Compare AARTM and AARTM+Q on a small synthetic collection')
    parser.add_argument('--documents', type=Path, default=Path(__file__).resolve().parents[1] / 'data/demo_documents.json')
    parser.add_argument('--topics', type=int, default=3)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'data/results/demo_comparison.json')
    args = parser.parse_args()
    documents = json.loads(args.documents.read_text(encoding='utf-8'))
    if len(documents) < 3 or not 1 <= args.topics <= len(documents) - len(documents[2::3]):
        parser.error('Need at least three documents and at most one topic per training document')
    evaluation_indices = list(range(2, len(documents), 3))
    training_indices = [i for i in range(len(documents)) if i not in evaluation_indices]
    training_documents = [documents[i] for i in training_indices]
    evaluation_documents = [documents[i] for i in evaluation_indices]
    prep = DatasetPreprocessor()
    training_tokens = prep.tokenize(training_documents)
    vocabulary = {word: i for i, word in enumerate(sorted({word for doc in training_tokens for word in doc}))}
    prep = DatasetPreprocessor(vocabulary=vocabulary)
    data, bounds = prep.fit_transform_tokenized(training_tokens)
    vocabulary = prep.vocabulary
    chunks = []
    for doc in training_documents:
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
    words = sorted(vocabulary, key=vocabulary.get)
    models = {}
    for name, initialization, kl_strength in [('AARTM', 0.0, 0.0), ('AARTM+Q', 0.95, 0.1)]:
        phi, n_t = fit_semantic_aartm(data, bounds, q, seed=args.seed, initialization=initialization, kl_strength=kl_strength, steps=args.steps)
        phi = np.asarray(phi)
        if not np.isfinite(phi).all() or (phi < 0).any() or not np.allclose(phi.sum(axis=1), 1, atol=1e-5):
            raise RuntimeError(f'{name}: invalid word-topic probabilities')
        beta = word_topic_probabilities(phi, data)
        ppl, hidden = completion_perplexity(beta, prep.tokenize(evaluation_documents), vocabulary)
        if not np.isfinite(ppl):
            raise RuntimeError(f'{name}: nonfinite perplexity')
        models[name] = dict(completion_ppl=ppl, held_out_tokens=hidden,
                            topics=[[words[i] for i in np.argsort(beta[:, t])[-10:][::-1]] for t in range(args.topics)])
    result = dict(purpose='synthetic comparison; excluded from paper results',
                  source_sha256=hashlib.sha256(args.documents.read_bytes()).hexdigest(),
                  training_indices=training_indices, evaluation_indices=evaluation_indices,
                  vocabulary=len(words), steps=args.steps, seed=args.seed, topic_count=args.topics,
                  encoder='sentence-transformers/all-MiniLM-L6-v2',
                  encoder_revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
                  evaluation='Every fifth preprocessed token is held out; only training-vocabulary words are scored; 40 inference steps',
                  models=models,
                  completion_ppl_reduction_percent=100 * (1 - models['AARTM+Q']['completion_ppl'] / models['AARTM']['completion_ppl']))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, phi=phi, q=q, n_t=np.asarray(n_t), words=np.array(words))


if __name__ == '__main__':
    main()

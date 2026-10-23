$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Path 'build' -Force | Out-Null
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -disable-installer -output-directory=build main.tex | Out-File -FilePath 'build/xelatex-first.txt' -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed; see docs/build/xelatex-first.txt' }
    & bibtex build/main | Out-File -FilePath 'build/bibtex.txt' -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed; see docs/build/bibtex.txt' }
    foreach ($pass in 1..2) {
        & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -disable-installer -output-directory=build main.tex | Out-File -FilePath "build/xelatex-final-$pass.txt" -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $pass failed" }
    }
    Copy-Item -LiteralPath 'build/main.pdf' -Destination 'main.pdf' -Force
    Write-Output 'Built docs/main.pdf'
}
finally {
    Pop-Location
}

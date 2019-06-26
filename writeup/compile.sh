pdflatex pnas-word-order-si.tex
bibtex pnas-word-order-si
pdflatex si-table-perrel-1.tex
pdflatex si-table-perrel-1a.tex
pdflatex si-table-perrel-1b.tex
pdflatex si-table-perrel-2a.tex
pdflatex si-table-perrel-3.tex

pdftoppm si-table-perrel-1.pdf si-table-perrel-1 -png
pdftoppm si-table-perrel-1a.pdf si-table-perrel-1a -png
pdftoppm si-table-perrel-1b.pdf si-table-perrel-1b -png
pdftoppm si-table-perrel-2a.pdf si-table-perrel-2a -png
pdftoppm si-table-perrel-3.pdf si-table-perrel-3 -png

pdflatex pnas-word-order-si.tex
bibtex pnas-word-order-si
pdflatex pnas-word-order-si.tex
pdflatex pnas-word-order-si.tex


pdftoppm ../results/dependency-length/figures/depl-violin-all.pdf depl-violin-all -png
pdftoppm ../results/dependency-length/figures/depLength-facet.pdf depLength-facet -png



pdflatex pnas-word-order-si.tex
bibtex pnas-word-order-si
pdflatex si-table-perrel-1.tex
pdflatex si-table-perrel-2.tex
pdflatex si-table-perrel-3.tex

pdftoppm si-table-perrel-1.pdf si-table-perrel-1 -png
pdftoppm si-table-perrel-2.pdf si-table-perrel-2 -png
pdftoppm si-table-perrel-3.pdf si-table-perrel-3 -png

pdflatex pnas-word-order-si.tex
bibtex pnas-word-order-si
pdflatex pnas-word-order-si.tex
pdflatex pnas-word-order-si.tex


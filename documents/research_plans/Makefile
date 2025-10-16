# original Makefile by Pasi "okun" Valminen, 2004
# edited by Pekka "PQ" Paalanen, 2004

# Modify the SRC and TARGET to your needs.

GS       := /usr/bin/gs
DVIPS    := /usr/bin/dvips
DVI2PDF  := /usr/bin/dvipdfm
TEXI2DVI := /usr/bin/texi2dvi

SRC      := $(wildcard *.tex) $(wildcard *.bib)
TARGET   := research_plan

GS       += -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite
DVIPS    += -q -R0 # allow running external programs (SECURITY!)
TEXI2DVI += -b

# http://www.cs.wisc.edu/~ghost/doc/AFPL/7.04/Ps2pdf.htm
# /screen, /printer, /prepress, /default
# there is also undocumented option: -c .setpdfwrite
#GS       += -dPDFSETTINGS=/screen


# texi2dvi runs bibtex automatically and latex as many times as needed.

all: $(TARGET).pdf

clean:
	rm -f *.aux *.bbl *.blg *.tmp *.dvi *.toc *.log *.out *~

distclean: clean
	rm -f $(TARGET).ps $(TARGET).pdf

view: $(TARGET).pdf
	acroread -openInNewWindow $(TARGET).pdf

Resegment_timetable_2011: Resegment_timetable_2011.csv Makefile
	csv2latex --noescape --nohead $@.csv | sed -f $@.sed > $@.tex

%.dvi: %.tex $(SRC) 
	$(TEXI2DVI) $<

%.ps: %.dvi
	$(DVIPS) -o $@ $(<:.tex=.dvi)

%.pdf: %.dvi
	$(DVI2PDF) $<

%.pdf: %.eps
	epstopdf --nogs $< | gs -q -sDEVICE=pdfwrite \
	-dAutoRotatePages=/None -dPDFSETTINGS=/printer \
	-dCompatibilityLevel=1.4 -dMaxSubsetPct=100 -dSubsetFonts=true \
	-dEmbedAllFonts=true -sOutputFile=$@ - -c quit

#%.pdf: %.ps
#	$(PS2PDF) $<

.SUFFIXES:
.PRECIOUS: %.dvi
.PHONY: all clean distclean view

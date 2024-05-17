# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_text_splitters.base import Language, TextSplitter


class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(
            self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = _split_text_with_regex(text, separator, self._keep_separator)
        _separator = "" if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator)

def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class DynamicSizeDynamicStepSplitter(TextSplitter):
    """Splitting text by recursively looking at characters and dynamically choosing separators."""

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            window_size: int = 100,
            step_size: int = 50,
            length_function: Callable[[str], int] = len,
            **kwargs: Any,

    ) -> None:
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", "。", "？", "！"]
        self._is_separator_regex = is_separator_regex
        self._window_size = window_size
        self._step_size = step_size
        self._keep_separator = keep_separator
        self._length_function = length_function

    def _split_text_with_regex(self, text: str, regex_separator: str) -> List[str]:
        """Helper method to split text using a regex separator."""
        if self._keep_separator:
            _splits = re.split(f"({regex_separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(regex_separator, text)
        return [s for s in splits if s != ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks by dynamically choosing the best separator."""
        final_chunks = [text]  # Start with the entire text as the only initial chunk

        for separator in separators:
            regex_separator = separator if self._is_separator_regex else re.escape(separator)
            new_chunks = []
            for chunk in final_chunks:
                if isinstance(chunk, str) and re.search(regex_separator, chunk):
                    splits = self._split_text_with_regex(chunk, regex_separator)
                    new_chunks.extend(splits)
                else:
                    if not isinstance(chunk, str):
                        print(f"Warning: Found non-string chunk: {chunk}")
            final_chunks = new_chunks

        # If no appropriate separator is found, use the whole text
        if not final_chunks:
            final_chunks = [text]

        # Further split the chunks based on window_size and step_size
        split_chunks = []
        for index, chunk in enumerate(final_chunks):
            start_index = index
            total_length = 0
            temp_chunk = ''
            while start_index < len(final_chunks):
                total_length += len(final_chunks[start_index])
                temp_chunk += final_chunks[start_index]
                if total_length >= self._window_size:
                    split_chunks.append(temp_chunk)
                    break
                start_index += 1

        # Remove chunks that do not exceed 100 characters in length
        split_chunks = [chunk for chunk in split_chunks if len(chunk) > 10]

        # Create list of tuples with chunk size and content
        chunk_sizes = [(len(chunk), chunk) for chunk in split_chunks]

        # Sort chunks by size in descending order
        chunk_sizes.sort(reverse=True, key=lambda x: x[0])
        print(chunk_sizes)
        return split_chunks

    def split_text(self, text: str) -> List[str]:
        """Public method to split text using configured separators."""
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
            cls, language: Language, **kwargs: Any
    ) -> DynamicSizeDynamicStepSplitter:
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        if language == Language.CPP:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.GO:
            return [
                # Split along function definitions
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JAVA:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.KOTLIN:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JS:
            return [
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PHP:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along class definitions
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PROTO:
            return [
                # Split along message definitions
                "\nmessage ",
                # Split along service definitions
                "\nservice ",
                # Split along enum definitions
                "\nenum ",
                # Split along option definitions
                "\noption ",
                # Split along import statements
                "\nimport ",
                # Split along syntax declarations
                "\nsyntax ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RST:
            return [
                # Split along section titles
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # Split along directive markers
                "\n\n.. *\n\n",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUBY:
            return [
                # Split along method definitions
                "\ndef ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUST:
            return [
                # Split along function definitions
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # Split along control flow statements
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SCALA:
            return [
                # Split along class definitions
                "\nclass ",
                "\nobject ",
                # Split along method definitions
                "\ndef ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SWIFT:
            return [
                # Split along function definitions
                "\nfunc ",
                # Split along class definitions
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LATEX:
            return [
                # First, try to split along Latex sections
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # Now split by environments
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # Now split by math environments
                "\n\\\begin{align}",
                "$$",
                "$",
                # Now split by the normal type of lines
                " ",
                "",
            ]
        elif language == Language.HTML:
            return [
                # First, try to split along HTML tags
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        elif language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # Split along class definitions
                "\nclass ",
                "\nabstract ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                # Split along control flow statements
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                # Split by exceptions
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SOL:
            return [
                # Split along compiler information definitions
                "\npragma ",
                "\nusing ",
                # Split along contract definitions
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # Split along method definitions
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.COBOL:
            return [
                # Split along divisions
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                # Split along sections within DATA DIVISION
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                # Split along sections within PROCEDURE DIVISION
                "\nINPUT-OUTPUT SECTION.",
                # Split along paragraphs and common statements
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                # Split by the normal type of lines
                "\n",
                " ",
                "",
            ]

        else:
            raise ValueError(
                f"Language {language} is not supported! "
                f"Please choose from {list(Language)}"
            )

if __name__ == '__main__':
    text ="Paper Info\n\nTitle: Bistability between π-diradical open-shell and closed-shell states in indeno[1,2-a]fluorene\nPublish Date: Unkown\nAuthor List: Shantanu Mishra (from IBM Research Europe -Zurich), Manuel Vilas-Varela (from Department of Organic Chemistry, Center for Research in Biological Chemistry and Molecular Materials (CiQUS), University of Santiago de Compostela), Leonard-Alexander Lieske (from IBM Research Europe -Zurich), Ricardo Ortiz (from Donostia International Physics Center (DIPC)), Igor Rončević (from Department of Chemistry, University of Oxford), Florian Albrecht (from IBM Research Europe -Zurich), Diego Peña (from Department of Organic Chemistry, Center for Research in Biological Chemistry and Molecular Materials (CiQUS), University of Santiago de Compostela), Leo Gross (from IBM Research Europe -Zurich)\n\nFigure\n\nFig. 1 | Non-benzenoid non-alternant polycyclic conjugated hydrocarbons.a, Classical nonbenzenoid non-alternant polycyclic conjugated hydrocarbons: pentalene, azulene and heptalene.b, Generation of indacenes and indenoindenes through benzinterposition and benzannelation of pentalene, respectively.Gray filled rings represent Clar sextets.c, Closed-shell Kekulé (left) and openshell non-Kekulé (right) resonance structures of QDMs.Note that meta-QDM is a non-Kekulé molecule.All indenofluorene isomers, being derived through benzannelation of indacenes, contain a central QDM moiety.d, Closed-shell Kekulé (top) and open-shell non-Kekulé (bottom) resonance structures of indenofluorenes.Compared to their closed-shell structures, 1 and 5 gain two Clar sextets in the openshell structure, while 2-4 gain only one Clar sextet in the open-shell structure.Colored bonds in d highlight the ortho-and para-QDM moieties in the two closed-shell Kekulé structures of 5. e, Scheme of on-surface generation of 5 by voltage pulse-induced dehydrogenation of 6 (C20H14).Structures 7 and 8 represent the two monoradical species (C20H13).\nFig. 2 | Characterization of open-shell indeno[1,2-a]fluorene on bilayer NaCl/Au(111).a, DFTcalculated wave functions of the frontier orbitals of 5OS in the triplet configuration for the spin up (occupied) level (isovalue: 0.002 e -Å -3 ).Blue and red colors represent opposite phases of the wave function.b, Corresponding DFT-calculated spin density of 5OS (isovalue: 0.01 e -Å -3).Blue and orange colors represent spin up and spin down densities, respectively.c, Probability density of the SOMOs of 5OS (isovalue: 0.001 e -Å -3 ).d, DFT-calculated bond lengths of 5OS.e, Constant-height I(V) spectra acquired on a species of 5 assigned as 5OS, along with the corresponding dI/dV(V) spectra.Open feedback parameters: V = -2 V, I = 0.17 pA (negative bias side) and V = 2 V, I = 0.17 pA (positive bias side).Acquisition position of the spectra is shown in Supplementary Fig.7.f, Scheme of many-body transitions associated to the measured ionic resonances of 5OS.Also shown are STM images of assigned 5OS at biases where the corresponding transitions become accessible.Scanning parameters: I = 0.3 pA (V = -1.2V and -1.5 V) and 0.2 pA (V = 1.3 V and 1.6 V). g, Laplace-filtered AFM image of assigned 5OS.STM set point: V = 0.2 V, I = 0.5 pA on bilayer NaCl, Δz = -0.3Å.The tip-height offset Δz for each panel is provided with respect to the STM setpoint, and positive (negative) values of Δz denote tip approach (retraction) from the STM setpoint.f and g show the same molecule at the same adsorption site, which is next to a trilayer NaCl island.The bright and dark features in the trilayer NaCl island in g correspond to Cl -and Na + ions, respectively.Scale bars: 10 Å (f) and 5 Å (g).\nFig. 3 | Characterization of closed-shell indeno[1,2-a]fluorene on bilayer NaCl/Au(111).a, DFTcalculated wave functions of the frontier orbitals of closed-shell 5 0 (isovalue: 0.002 e -Å -3 ).The wave functions shown here are calculated for the 5para geometry.b, DFT-calculated bond lengths of 5ortho (top) and 5para (bottom).c, Constant-height I(V) spectra acquired on a species of 5 assigned as 5para, along with the corresponding dI/dV(V) spectra.Open feedback parameters: V = -2 V, I = 0.15 pA (negative bias side) and V = 2.2 V, I = 0.15 pA (positive bias side).Acquisition position of the spectra is shown in Supplementary Fig. 7. d, Scheme of many-body transitions associated to the measured ionic resonances of 5para.Also shown are STM images of assigned 5para at biases where the corresponding transitions become accessible.Scanning parameters: I = 0.15 pA (V = -1.5 V) and 0.2 pA (V = 1.7 V). e, Laplace-filtered AFM image of assigned 5para.STM set point: V = 0.2 V, I = 0.5 pA on bilayer NaCl, Δz = -0.7 Å. f, Selected bonds labeled for highlighting bond order differences between 5para and 5ortho.For the bond pairs a/b, c/d and e/f, the bonds labeled in bold exhibit a higher bond order than their neighboring labeled bonds in 5para.g, Laplace-filtered AFM images of 5 on bilayer NaCl/Cu(111) showing switching between 5OS and 5para as the molecule changes its adsorption position.The faint protrusion adjacent to 5 is a defect that stabilizes the adsorption of 5. STM set point: V = 0.2 V, I = 0.5 pA on bilayer NaCl, Δz = -0.3Å. STM and STS data in c and d are acquired on the same species, while the AFM data in e is acquired on a different species.Scale bars: 10 Å (d) and 5 Å (e,g).\nNMR (300 MHz, CDCl3) δ: 7.51 (m, 2H), 7.40 -7.28 (m, 5H), 7.27 -7.20 (m, 2H), 7.13 (d, J = 7.7 Hz, 1H), 2.07 (s, 3H), 1.77 (s, 3H) ppm. 13C NMR-DEPT (75 MHz, CDCl3, 1:1 mixture of atropisomers) δ: 141.2 (C), 141.1 (C), 140.0 (C), 139.4 (2C), 137.5 (C), 137.4 (C), 136.0 (3C), 134.8 (C), 134.5 (C), 134.1 (C), 134.0 (C), 133.7 (C), 133.6 (C), 131.6 (CH), 131.2 (CH), 131.1 (CH), 130.7 (CH), 129.8 (CH), 129.7 (CH), 129.5 (CH), 129.4 (CH), 129.0 (CH), 128.9 (CH), 128.7 (2CH), 128.6 (2CH), 127.2 (CH), 127.1 (CH), 127.0 (CH), 126.9 (CH), 126.7 (CH), 126.6 (CH), 20.6 (CH3), 20.5 (CH3), 17.7 (CH3), 17.5 (CH3) ppm.MS (APCI) m/z (%): 327 (M+1, 100).HRMS: C20H16Cl2; calculated: 327.0702, found: 327.0709.\nNMR (500 MHz, CDCl3) δ: 7.93 (d, J = 7.6 Hz, 1H), 7.85 (d, J = 7.5 Hz, 1H), 7.78 (d, J = 7.7 Hz, 1H), 7.65 (d, J = 7.4 Hz, 1H), 7.61 (d, J = 7.5 Hz, 1H), 7.59 (d, J = 7.7 Hz, 1H), 7.47 (ddd, J = 8.4, 7.2, 1.1 Hz, 1H), 7.42 (dd, J = 8.1, 7.0 Hz, 1H), 7.35 (m, 2H), 4.22 (s, 3H), 4.02 (s, 3H).ppm. 13C NMR-DEPT (125 MHz, CDCl3) δ: 144.1 (C), 143.3 (C), 142.3 (C), 141.9 (C), 141.8 (C), 141.2 (C), 138.2 (C), 136.5 (C), 127.0 (CH), 126.9 (CH), 126.7 (CH), 126.6 (CH), 125.3 (CH), 125.2 (CH), 123.6 (CH), 122.2 (CH), 119.9 (CH), 118.4 (CH), 37.4 (CH2), 36.3 (CH2).ppm.MS (APCI) m/z (%): 254 (M+, 88).HRMS: C20H14; calculated: 254.1090, found: 254.1090.\n\nabstract\n\nIndenofluorenes are non-benzenoid conjugated hydrocarbons that have received great interest owing to their unusual electronic structure and potential applications in nonlinear optics and photovoltaics. Here, we report the generation of unsubstituted indeno[1,2-a]fluorene, the final and yet unreported parent indenofluorene regioisomer, on various surfaces by cleavage of two C-H bonds in 7,12-dihydro indeno[1,2-a]fluorene through voltage pulses applied by the tip of a combined scanning tunneling microscope and atomic force microscope.\nOn bilayer NaCl on Au(111), indeno[1,2a]fluorene is in the neutral charge state, while it exhibits charge bistability between neutral and anionic states on the lower work function surfaces of bilayer NaCl on Ag(111) and Cu(111). In the neutral state, indeno[1,2-a]fluorene exhibits either of two ground states: an open-shell π-diradical state, predicted to be a triplet by density functional and multireference many-body perturbation theory calculations, or a closedshell state with a para-quinodimethane moiety in the as-indacene core.\nSwitching between open-and closed-shell states of a single molecule is observed by changing its adsorption site on NaCl. The inclusion of non-benzenoid carbocyclic rings is a viable route to tune the physicochemical properties of polycyclic conjugated hydrocarbons (PCHs) . Non-benzenoid polycycles may lead to local changes in strain, conjugation, aromaticity, and, relevant to the context of the present work, induce an open-shell ground state of the corresponding PCHs .\nMany nonbenzenoid PCHs are also non-alternant, where the presence of odd-membered polycycles breaks the bipartite symmetry of the molecular network . Figure shows classical examples of non-benzenoid non-alternant PCHs, namely, pentalene, azulene and heptalene. Whereas azulene is a stable PCH exhibiting Hückel aromaticity ([4n+2] π-electrons, n = 2), pentalene and heptalene are unstable Hückel antiaromatic compounds with [4n] π-electrons, n = 2 (pentalene) and n = 3 (heptalene).\nBenzinterposition of pentalene generates indacenes, consisting of two isomers s-indacene and as-indacene (Fig. ). Apart from being antiaromatic, indacenes also contain proaromatic quinodimethane (QDM) moieties (Fig. ) , which endows them with potential open-shell character. While the parent s-indacene and asindacene have never been isolated, stable derivatives of s-indacene bearing bulky substituents have been synthesized .\nA feasible strategy to isolate congeners of otherwise unstable non-benzenoid non-alternant PCHs is through fusion of benzenoid rings at the ends of the π-system, that is, benzannelation. For example, while the parent pentalene is unstable, the benzannelated congener indeno[2,1-a]indene is stable under ambient conditions (Fig. ) .\nHowever, the position of benzannelation is crucial for stability: although indeno[2,1a]indene is stable, its regioisomer indeno[1,2-a]indene (Fig. ) oxidizes under ambient conditions . Similarly, benzannelation of indacenes gives rise to the family of PCHs known as indenofluorenes (Fig. ), which constitute the topic of the present work.\nDepending on the benzannelation position and the indacene core, five regioisomers can be constructed, namely, indeno [ Practical interest in indenofluorenes stems from their low frontier orbital gap and excellent electrochemical characteristics that render them as useful components in organic electronic devices .\nThe potential open-shell character of indenofluorenes has led to several theoretical studies on their use as non-linear optical materials and as candidates for singlet fission in organic photovoltaics . Recent theoretical work has also shown that indenofluorene-based ladder polymers may exhibit fractionalized excitations.\nFundamentally, indenofluorenes represent model systems to study the interplay between aromaticity and magnetism at the molecular scale . Motivated by many of these prospects, the last decade has witnessed intensive synthetic efforts toward the realization of indenofluorenes. Derivatives of 1-4 have been realized in solution , while 1-3 have also been synthesized on surfaces and characterized using scanning tunneling microscopy (STM) and atomic force microscopy (AFM), which provide information on molecular orbital densities , molecular structure and oxidation state .\nWith regards to the open-shell character of indenofluorenes, 2-4 are theoretically and experimentally interpreted to be closed-shell, while calculations indicate that 1 and 5 should exhibit open-shell ground states . Bulk characterization of mesitylsubstituted 1, including X-ray crystallography, temperature-dependent NMR, and electron spin resonance spectroscopy, provided indications of its open-shell ground state .\nElectronic characterization of 1 on Au(111) surface using scanning tunneling spectroscopy (STS) revealed a low electronic gap of 0.4 eV (ref. ). However, no experimental proof of an openshell ground state of 1 on Au(111), such as detection of singly occupied molecular orbitals (SOMOs) or spin excitations and correlations due to unpaired electrons , was shown.\nIn this work, we report the generation and characterization of unsubstituted 5. Our research is motivated by theoretical calculations that indicate 5 to exhibit the largest diradical character among all indenofluorene isomers . The same calculations also predict that 5 should possess a triplet ground state.\nTherefore, 5 would qualify as a Kekulé triplet, of which only a handful of examples exist . However, definitive synthesis of 5 has never been reported so far. Previously, Dressler et al. reported transient isolation of mesityl-substituted 5, where it decomposed both in the solution and in solid state , and only the structural proof of the corresponding dianion was obtained.\nOn-surface generation of a derivative of 5, starting from truxene as a precursor, was recently reported . STM data on this compound, containing the indeno[1,2-a]fluorene moiety as part of a larger PCH, was interpreted to indicate its open-shell ground state. However, the results did not imply the ground state of unsubstituted 5. Here, we show that on insulating surfaces 5 can exhibit either of two ground states: an open-shell or a closed-shell.\nWe infer the existence of these two ground states based on high-resolution AFM imaging with bond-order discrimination and STM imaging of molecular orbital densities . AFM imaging reveals molecules with two different geometries. Characteristic bond-order differences in the two geometries concur with the geometry of either an open-or a closed-shell state.\nConcurrently, STM images at ionic resonances show molecular orbital densities corresponding to SOMOs for the open-shell geometry, but orbital densities of the highest occupied molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO) for the closed-shell geometry. Our experimental results are in good agreement with density functional theory (DFT) and multireference perturbation theory calculations.\nFinally, we observe switching between open-and closed-shell states of a single molecule by changing its adsorption site on the surface. Synthetic strategy toward indeno[1,2-a]fluorene. The generation of 5 relies on the solution-phase synthesis of the precursor 7,12-dihydro indeno[1,2-a]fluorene (6). Details on synthesis and characterization of 6 are reported in Supplementary Figs.\n. Single molecules of 6 are deposited on coinage metal (Au(111), Ag(111) and Cu(111)) or insulator surfaces. In our work, insulating surfaces correspond to two monolayer-thick (denoted as bilayer) NaCl on coinage metal surfaces. Voltage pulses ranging between 4-6 V are applied by the tip of a combined STM/AFM system, which result in cleavage of one C-H bond at each of the pentagonal apices of 6, thereby leading to the generation of 5 (Fig. ).\nIn the main text, we focus on the generation and characterization of 5 on insulating surfaces. Generation and characterization of 5 on coinage metal surfaces is shown in Supplementary Fig. . ). Blue and orange colors represent spin up and spin down densities, respectively. c, Probability density of the SOMOs of 5OS (isovalue: 0.001 e -Å -3 ).\nd, DFT-calculated bond lengths of 5OS. e, Constant-height I(V) spectra acquired on a species of 5 assigned as 5OS, along with the corresponding dI/dV(V) spectra. Open feedback parameters: V = -2 V, I = 0.17 pA (negative bias side) and V = 2 V, I = 0.17 pA (positive bias side). Acquisition position of the spectra is shown in Supplementary Fig. . f, Scheme of many-body transitions associated to the measured ionic resonances of 5OS.\nAlso shown are STM images of assigned 5OS at biases where the corresponding transitions become accessible. Scanning parameters: I = 0.3 pA (V = -1.2 V and -1.5 V) and 0.2 pA (V = 1.3 V and 1.6 V). g, Laplace-filtered AFM image of assigned 5OS. STM set point: V = 0.2 V, I = 0.5 pA on bilayer NaCl, Δz = -0.3\nÅ. The tip-height offset Δz for each panel is provided with respect to the STM setpoint, and positive (negative) values of Δz denote tip approach (retraction) from the STM setpoint. f and g show the same molecule at the same adsorption site, which is next to a trilayer NaCl island. The bright and dark features in the trilayer NaCl island in g correspond to Cl -and Na + ions, respectively.\nScale bars: 10 Å (f) and 5 Å (g). To experimentally explore the electronic structure of 5, we used bilayer NaCl films on coinage metal surfaces to electronically decouple the molecule from the metal surfaces. Before presenting the experimental findings, we summarize the results of our theoretical calculations performed on 5 in the neutral charge state (denoted as 5 0 ).\nWe start by performing DFT calculations on 5 0 in the gas phase. Geometry optimization performed at the spin-unrestricted UB3LYP/6-31G level of theory leads to one local minimum, 5OS, the geometry of which corresponds to the open-shell resonance structure of 5 (Fig. , the label OS denotes open-shell).\nThe triplet electronic configuration of 5OS is the lowest-energy state, with the openshell singlet configuration 90 meV higher in energy. Geometry optimization performed at the restricted closed-shell RB3LYP/6-31G level reveals two local minima, 5para and 5ortho, the geometries of which (Fig. ) exhibit bond length alternations in line with the presence of a para-or an ortho-QDM moiety, respectively, in the as-indacene core of the closed-shell resonance structures of 5 (Fig. ) .\nRelative to 5OS in the triplet configuration, 5para and 5ortho are 0.40 and 0.43 eV higher in energy, respectively. Additional DFT results are shown in Supplementary Fig. . To gain more accurate insights into the theoretical electronic structure of 5, we performed multireference perturbation theory calculations (Supplementary Fig. ) based on quasi-degenerate second-order n-electron valence state perturbation theory (QD-NEVPT2).\nIn so far as the order of the ground and excited states are concerned, the results of QD-NEVPT2 calculations qualitatively match with DFT calculations. For 5OS, the triplet configuration remains the lowest-energy state, with the open-shell singlet configuration 60 meV higher in energy. The energy differences between the open-and closed-shell states are substantially reduced in QD-NEVPT2 calculations, with 5para and 5ortho only 0.11 and 0.21 eV higher in energy, respectively, compared to 5OS in the triplet configuration.\nWe also performed nucleus-independent chemical shift calculations to probe local aromaticity of 5 in the openand closed-shell states. While 5OS in the triplet configuration exhibits local aromaticity at the terminal benzenoid rings, 5OS in the open-shell singlet configuration, 5para and 5ortho all display antiaromaticity (Supplementary Fig. ).\nThe choice of the insulating surface determines the charge state of 5: while 5 adopts neutral charge state on the high work function bilayer NaCl/Au(111) surface (irrespective of its openor closed-shell state, Supplementary Fig. ), 5 exhibits charge bistability between 5 0 and the anionic state 5 -1 on the lower work function bilayer NaCl/Ag(111) and Cu(111) surfaces (Supplementary Figs. ).\nIn the main text, we focus on the characterization of 5 on bilayer NaCl/Au(111). Characterization of charge bistable 5 is reported in Supplementary Figs. . We first describe experiments on 5 on bilayer NaCl/Au(111), where 5 exhibits a geometry corresponding to the calculated 5OS geometry, and an open-shell electronic configuration.\nWe compare the experimental data on this species to calculations on 5OS with a triplet configuration, as theory predicts a triplet ground state for 5OS. For 5OS, the calculated frontier orbitals correspond to the SOMOs ψ1 and ψ2 (Fig. ), whose spin up levels are occupied and the spin down levels are empty.\nFigure shows the DFT-calculated bond lengths of 5OS, where the two salient features, namely, the small difference in the bond lengths within each ring and the notably longer bond lengths in the pentagonal rings, agree with the open-shell resonance structure of 5 (Fig. ). Figure shows an AFM image of 5 adsorbed on bilayer NaCl/Au(111) that we assign as 5OS, where the bond-order differences qualitatively correspond to the calculated 5OS geometry (discussed and compared to the closed-shell state below).\nDifferential conductance spectra (dI/dV(V), where I and V denote the tunneling current and bias voltage, respectively) acquired on assigned 5OS exhibit two peaks centered at -1.5 V and 1.6 V (Fig. ), which we assign to the positive and negative ion resonances (PIR and NIR), respectively. Figure shows the corresponding STM images acquired at the onset (V = -1.2\nV/1.3 V) and the peak (V = -1.5 V/1.6 V) of the ionic resonances. To draw a correspondence between the STM images and the molecular orbital densities, we consider tunneling events as many-body electronic transitions between different charge states of 5OS (Fig. ). Within this framework, the PIR corresponds to transitions between 5 0 and the cationic state 5 .\nAt the onset of the PIR at -1.2 V, an electron can only be detached from the SOMO ψ1 and the corresponding STM image at -1.2 V shows the orbital density of ψ1. Increasing the bias to the peak of the PIR at -1.5 V, it becomes possible to also empty the SOMO ψ2, such that the corresponding STM image shows the superposition of ψ1 and ψ2, that is, |ψ1| 2 + |ψ2| 2 (ref.\n). Similarly, the NIR corresponds to transitions between 5 0 and 5 -1 . At the NIR onset of 1.3 V, only electron attachment to ψ2 is energetically possible. At 1.6 V, electron attachment to ψ1 also becomes possible, and the corresponding STM image shows the superposition of ψ1 and ψ2. The observation of the orbital densities of SOMOs, and not the hybridized HOMO and LUMO, proves the open-shell ground state of assigned 5OS.\nMeasurements of the monoradical species with a doublet ground state are shown in Supplementary Fig. . Unexpectedly, another species of 5 was also experimentally observed that exhibited a closedshell ground state. In contrast to 5OS, where the frontier orbitals correspond to the SOMOs ψ1 and ψ2, DFT calculations predict orbitals of different shapes and symmetries for 5para and 5ortho, denoted as α and β and shown in Fig. .\nFor 5ortho, α and β correspond to HOMO and LUMO, respectively. The orbitals are inverted in energy and occupation for 5para, where β is the HOMO and α is the LUMO. Fig. shows an AFM image of 5 that we assign as 5para. We experimentally infer its closed-shell state first by using qualitative bond order discrimination by AFM.\nIn high-resolution AFM imaging, chemical bonds with higher bond order are imaged brighter (that is, with higher frequency shift Δf) due to stronger repulsive forces, and they appear shorter . In Fig. , we label seven bonds whose bond orders show significant qualitative differences in the calculated 5ortho, 5para (Fig. ) and 5OS (Fig. ) geometries.\nIn 5para, the bonds b and d exhibit a higher bond order than a and c, respectively. This pattern is reversed for 5ortho, while the bond orders of the bonds a-d are all similar and small for 5OS. Furthermore, in 5para bond f exhibits a higher bond order than e, while in 5ortho and 5OS bonds e and f exhibit similar bond order (because they belong to Clar sextets).\nFinally, the bond labeled g shows a higher bond order in 5para than in 5ortho and 5OS. The AFM image of assigned 5para shown in Fig. indicates higher bond orders of the bonds b, d and f compared to a, c and e, respectively. In addition, the bond g appears almost point-like and with enhanced Δf contrast compared to its neighboring bonds, indicative of a high bond order (see Supplementary Fig. for height-dependent measurements).\nThese observations concur with the calculated 5para geometry (Fig. ). Importantly, all these distinguishing bond-order differences are distinctly different in the AFM image of 5OS shown in Fig. , which is consistent with the calculated 5OS geometry (Fig. ). In the AFM images of 5OS (Fig. and Supplementary Fig. ), the bonds a-d at the pentagon apices appear with similar contrast and apparent bond length.\nThe bonds e and f at one of the terminal benzenoid rings also exhibit similar contrast and apparent bond length, while the central bond g appears longer compared to assigned 5para. Further compelling evidence for the closed-shell state of assigned 5para is obtained by STM and STS. dI/dV(V) spectra acquired on an assigned 5para species exhibit two peaks centered at -1.4 V (PIR) and 1.6 V (NIR) (Fig. ).\nSTM images acquired at these biases (Fig. ) show the orbital densities of β (-1.4 V) and α (1.6 V). First, the observation of α and β as the frontier orbitals of this species, and not the SOMOs, strongly indicates its closed-shell state. Second, consistent with AFM measurements that indicate good correspondence to the calculated 5para geometry, we observe β as the HOMO and α as the LUMO.\nFor 5ortho, α should be observed as the HOMO and β as the LUMO. We did not observe molecules with the signatures of 5ortho in our experiments. We observed molecules in open-(5OS, Fig. ) and closed-shell (5para, Fig. ) states in similar occurrence after their generation from 6 on the surface. We could also switch individual molecules between open-and closed-shell states as shown in Fig. and Supplementary Fig. .\nTo this end, a change in the adsorption site of a molecule was induced by STM imaging at ionic resonances, which often resulted in movement of the molecule. The example presented in Fig. shows a molecule that was switched from 5para to 5OS and back to 5para. The switching is not directed, that is, we cannot choose which of the two species will be formed when changing the adsorption site, and we observed 5OS and 5para in approximately equal yields upon changing the adsorption site.\nThe molecule in Fig. is adsorbed on top of a defect that stabilizes its adsorption geometry on bilayer NaCl. At defect-free adsorption sites on bilayer NaCl, that is, without a third layer NaCl island or atomic defects in the vicinity of the molecule, 5 could be stably imaged neither by AFM nor by STM at ionic resonances (Supplementary Fig. ).\nWithout changing the adsorption site, the state of 5 (open-or closedshell) never changed, including the experiments on bilayer NaCl/Ag(111) and Cu(111), on which the charge state of 5 could be switched (Supplementary Figs. ). Also on these lower work function surfaces, both open-and closed-shell species were observed for 5 0 and both showed charge bistability between 5 0 (5OS or 5para) and 5 -1 (Supplementary Figs. ).\nThe geometrical structure of 5 -1 probed by AFM, and its electronic structure probed by STM imaging at the NIR (corresponding to transitions between 5 -1 and the dianionic state 5 -2 ), are identical within the measurement accuracy for the charged species of both 5OS and 5para. When cycling the charge state of 5 between 5 0 and 5 -1 several times, we always observed the same state (5OS or 5para) when returning to 5 0 , provided the molecule did not move during the charging/discharging process.\nBased on our experimental observations we conclude that indeno[1,2-a]fluorene (5), the last unknown indenofluorene isomer, can be stabilized in and switched between an open-shell (5OS) and a closed-shell (5para) state on NaCl. For the former, both DFT and QD-NEVPT2 calculations predict a triplet electronic configuration.\nTherefore, 5 can be considered to exhibit the spin-crossover effect, involving magnetic switching between high-spin (5OS) and low-spin (5para) states, coupled with a reversible structural transformation. So far, the spin-crossover effect has mainly only been observed in transition-metal-based coordination compounds with a near-octahedral geometry .\nThe observation that the switching between open-and closedshell states is related to changes in the adsorption site but is not achieved by charge-state cycling alone, indicates that the NaCl surface and local defects facilitate different electronic configurations of 5 depending on the adsorption site.\nGas-phase QD-NEVPT2 calculations predict that 5OS is the ground state, and the closed-shell 5para and 5ortho states are 0.11 and 0.21 eV higher in energy. The experiments, showing bidirectional switching between 5OS and 5para, indicate that a change in the adsorption site can induce sufficient change in the geometry of 5 (leading to a corresponding change in the ground state electronic configuration) and thus induce switching.\nSwitching between open-and closed-shell states in 5 does not require the breaking or formation of covalent bonds , but a change of adsorption site on NaCl where the molecule is physisorbed. Our results should have implications for single-molecule devices, capitalizing on the altered electronic and chemical properties of a system in π-diradical open-shell and closed-shell states such as frontier orbital and singlet-triplet gaps, and chemical reactivity.\nFor possible future applications as a single-molecule switch, it might be possible to also switch between open-and closed-shell states by changing the local electric field, such as by using chargeable adsorbates . Scanning probe microscopy measurements and sample preparation. STM and AFM measurements were performed in a home-built system operating at base pressures below 1×10 -10 mbar and a base temperature of 5 K. Bias voltages are provided with respect to the sample.\nAll STM, AFM and spectroscopy measurements were performed with carbon monoxide (CO) functionalized tips. AFM measurements were performed in non-contact mode with a qPlus sensor . The sensor was operated in frequency modulation mode with a constant oscillation amplitude of 0.5 Å. STM measurements were performed in constantcurrent mode, AFM measurements were performed in constant-height mode with V = 0 V, and I(V) and Δf(V) spectra were acquired in constant-height mode.\nPositive (negative) values of the tip-height offset Δz represent tip approach (retraction) from the STM setpoint. All dI/dV(V) spectra are obtained by numerical differentiation of the corresponding I(V) spectra. STM and AFM images, and spectroscopy curves, were post-processed using Gaussian low-pass filters.\nAu(111), Ag(111) and Cu(111) surfaces were cleaned by iterative cycles of sputtering with Ne + ions and annealing up to 800 K. NaCl was thermally evaporated on Au(111), Ag(111) and Cu(111) surfaces held at 323 K, 303 K and 283 K, respectively. This protocol results in the growth of predominantly bilayer (100)-terminated islands, with a minority of trilayer islands.\nSub-monolayer coverage of 6 on surfaces was obtained by flashing an oxidized silicon wafer containing the precursor molecules in front of the cold sample in the microscope. CO molecules for tip functionalization were dosed from the gas phase on the cold sample. Density functional theory calculations. DFT was employed using the PSI4 program package .\nAll molecules with different charge (neutral and anionic) and electronic (open-and closed-shell) states were independently investigated in the gas phase. The B3LYP exchangecorrelation functional with 6-31G basis set was employed for structural relaxation and singlepoint energy calculations. The convergence criteria were set to 10 −4 eV Å −1 for the total forces and 10 −6 eV for the total energies.\nMultireference calculations. Multireference calculations were performed on the DFToptimized geometries using the QD-NEVPT2 level of theory , with three singlet roots and one triplet root included in the state-averaged calculation. A (10,10) active space (that is, 10 electrons in 10 orbitals) was used along with the def2-TZVP basis set .\nIncreasing either the active space size or expanding the basis set resulted in changes of about 50 meV for relative energies of the singlet and triplet states. These calculations were performed using the ORCA program package . Nucleus-independent chemical shift (NICS) calculations. Isotropic nucleus-independent chemical shift values were evaluated at the centre of each ring using the B3LYP exchangecorrelation functional with def2-TZVP basis set using the Gaussian 16 software package .\nStarting materials (reagent grade) were purchased from TCI and Sigma-Aldrich and used without further purification. Reactions were carried out in flame-dried glassware and under an inert atmosphere of purified Ar using Schlenk techniques. Thin-layer chromatography (TLC) was performed on Silica Gel 60 F-254 plates (Merck).\nColumn chromatography was performed on silica gel (40-60 µm). Nuclear magnetic resonance (NMR) spectra were recorded on a Bruker Varian Mercury 300 or Bruker Varian Inova 500 spectrometers. Mass spectrometry (MS) data were recorded in a Bruker Micro-TOF spectrometer. The synthesis of compound 6 was developed following the two-step synthetic route shown in Supplementary Fig. , which is based on the preparation of methylene-bridge polyarenes by means of Pd-catalyzed activation of benzylic C-H bonds .\nSupplementary Figure | Synthetic route to obtain compound 6. The complex Pd2(dba)3 (20 mg, 0.02 mmol) was added over a deoxygenated mixture of 1,3-dibromo-2,4-dimethylbenzene (9, 100 mg, 0.38 mmol), boronic acid 10 (178 mg, 1.14 mmol), K2CO3 (314 mg, 2.28 mmol) and XPhos (35 mg, 0.08 mmol) in toluene (1:1, 10 mL), and the resulting mixture was heated at 90 °C for 2 h.\nAfter cooling to room temperature, the solvents were evaporated under reduced pressure. The reaction crude was purified by column chromatography (SiO2; hexane:CH2Cl2 9:1) affording 11 (94 mg, 76%) as a colorless oil. The complex Pd(OAc)2 (7 mg, 0.03 mmol) was added over a deoxygenated mixture of terphenyl 11 (90 mg, 0.27 mmol), K2CO3 (114 mg, 0.83 mmol) and ligand L (26 mg, 0.06 mmol) in NMP (2 mL).\nThe resulting mixture was heated at 160 °C for 4 h. After cooling to room temperature, H2O (30 mL) was added, and the mixture was extracted with EtOAc (3x15 mL). The combined organic extracts were dried over anhydrous Na2SO4, filtered, and evaporated under reduced pressure. The reaction crude was purified by column chromatography (SiO2; hexane:CH2Cl2 9:1) affording compound 6 (8 mg, 11%) as a white solid. in AFM imaging due to their reduced adsorption height compared to the rest of the carbon atoms.\nWe attribute this observation to the significantly different lattice parameter of Cu(111) (2.57 Å) compared to Au(111) and Ag(111) (2.95 Å and 2.94 Å, respectively) , such that the apical carbon atoms of the pentagonal rings of 5 adsorb on the on-top atomic sites on Au(111) and Ag(111), but not on Cu(111).\nOur speculation is based on a previous study of polymers of 1 on Au(111) by Di Giovannantonio et al. , where both tilted and planar individual units of 1 were observed depending on whether the apical carbon atoms of the pentagonal rings in 1 adsorbed on the on-top or hollow sites of the surface, respectively.\nGiven the strong molecule-metal interaction, we found no electronic state signatures of 5 on all three metal surfaces. STM set point for AFM images: V = 0. e, Frontier orbital spectrum of 5 -1 . In the anionic state, ψ2 becomes doubly occupied and ψ1 is the SOMO. Filled and empty circles denote occupied and empty orbitals, respectively.\nFor each panel, zero of the energy axis has been aligned to the respective highest-energy occupied orbital."
    text_splitter = DynamicSizeDynamicStepSplitter(window_size=500, step_size=1)
    all_splits = text_splitter.split_text(text)
    # print(all_splits)

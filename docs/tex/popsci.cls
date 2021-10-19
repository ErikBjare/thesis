% popsci.cls
% Magnus Hultin, 2016
% Flavius Gruian, 2016-06-23

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{popsci}[2016/06/23  Popular science paper, Department of Computer Science, Lund University]

\LoadClass[a4paper, 11pt, twocolumn]{article}
% use showframe in parameters if you want to check the margins
\RequirePackage[left=2cm,right=2cm,top=3cm,bottom=4cm, bindingoffset=0cm,ignorefoot]{geometry}

%\textheight = 600pt %540pt%592pt

\RequirePackage{ifxetex}

\ifxetex
  \usepackage{fontspec}
\else
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{lmodern}
\fi


\date{}



\RequirePackage{color}
\RequirePackage{colortbl}

\definecolor{LUorange}{RGB}{167,102,9}

\definecolor{gray}{RGB}{120,120,120} 


\setlength{\arrayrulewidth}{3pt}
\arrayrulecolor{gray}
\def\arraystretch{1.2}



% fancy headers, footers
\RequirePackage{fancyhdr}


\pagestyle{fancy}

\newcommand{\supervisor}[1]{\def \@supervisor {#1}}
\newcommand{\supervisors}[2]{\def \@firstsupervisor {#1}  \def \@secondsupervisor {#2}}
\newcommand{\examiner}[1]{\def \@examiner {#1}}
\newcommand{\student}[1]{\def \@student {#1} \def \@author {#1}}
\newcommand{\students}[2]{\def \@firststudent {#1} \def \@secondstudent {#2} \def \@author {#1, #2}}

\newcommand{\examensarbete}[1]{\def \@examensarbete {#1}}

\newcommand{\examensarbeteTwoRows}[2]{\def \@firstexamensarbete{#1} \def \@secondexamensarbete{#2}}


\newcommand{\presentationsdag}[1]{\def \@presentationsdag{#1}}

\presentationsdag{}
\examensarbete{}
\examensarbeteTwoRows{}{}
\student{}
\students{}{}

\RequirePackage{titling}


\lhead{
\vskip -3em
\begin{tabular}{| l }
  \sffamily \small INSTITUTIONEN FÖR DATAVETENSKAP \textbar\ LUNDS TEKNISKA HÖGSKOLA
  \ifx \@presentationsdag \empty 
  \else
    \textbar\ PRESENTERAD \@presentationsdag 
  \fi
  \\
  \\
  \ifx\@examensarbete \empty  
	\underline{\sffamily \textbf{EXAMENSARBETE} \@firstexamensarbete} \\
	\underline{\sffamily \@secondexamensarbete} \\
  \else 
  	\underline{\sffamily \textbf{EXAMENSARBETE} \@examensarbete} \\
  \fi
  \ifx \@student \empty 
    \underline{\sffamily \textbf{STUDENTER} \@firststudent, \@secondstudent}\\
  \else
    \underline{\sffamily \textbf{STUDENT} \@student}\\
  \fi
  \underline{\sffamily \textbf{HANDLEDARE} \@supervisor} \\
  \underline{\sffamily \textbf{EXAMINATOR} \@examiner}
\end{tabular}
}
\chead{}
\rhead{}


% Footers
\lfoot{}
\cfoot{}
\rfoot{}


\renewcommand{\headrulewidth}{0.0pt} % No header rule
\renewcommand{\footrulewidth}{0.0pt} % Thin footer rule


\makeatletter
% Title page

\newcommand{\HorRule}{\color{black} \rule{\linewidth}{1pt}} % Defines the gold horizontal rule around the title

\pretitle{\vspace{50pt} \begin{flushleft} \fontsize{25pt}{20} \usefont{OT1}{phv}{b}{n} \color{LUorange} \selectfont} % Horizontal rule before the title

\posttitle{\par\end{flushleft}\vskip 0.2em\HorRule} % Whitespace under the title

\preauthor{\begin{flushleft}\large \lineskip -0.2em \usefont{OT1}{phv}{m}{n} \color{black} \textsf{POPULÄRVETENSKAPLIG SAMMANFATTNING} \bfseries} % Author font configuration

\postauthor{\footnotesize \usefont{OT1}{phv}{m}{sl} \color{black} \par\end{flushleft} \vskip -1 em  \HorRule \vskip -4em} % Horizontal rule after the title


\newcommand{\theabstract}[1]{
%\vskip -35em

\twocolumn[
  \begin{@twocolumnfalse} 

\maketitle
\thispagestyle{fancy}

\renewcommand{\abstractname}{\vspace{-\baselineskip}}


\fontsize{14}{16}  \sffamily \selectfont \noindent #1 \vskip 1 em


 \end{@twocolumnfalse}
]


}


\ifx \@theabstract \@empty
		\ClassWarning{cslthse-msc}{The abstract is missing. Use \protect\theabstract{text}.}
\fi
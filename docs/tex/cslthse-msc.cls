% cslthse-msc.cls
% Copyright 2013-2020 Flavius Gruian <flavius.gruian@cs.lth.se>
% 
% Licensed under the GNU GPL version 3.0 or later.
% See http://www.gnu.org/licenses/gpl.html
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cslthse-msc}
		[2020/09/18 v1.6 
		Masters Thesis Class, Department of Computer Science, Lund University]

%\ExecuteOptions{12pt,a4paper,onecolumn, twoside, openright}
%\ProcessOptions\relax
%\LoadClass{report}
\LoadClass[12pt,a4paper,onecolumn,twoside,openright]{report}

% margins
% use showframe in parameters if you want to check the margins
\RequirePackage[left=2.5cm,right=2.5cm,top=3cm,bottom=3cm, bindingoffset=1cm]{geometry}

% Fonts! needs XeTeX
\RequirePackage{ifxetex}
\makeatletter
% for textbullets
\usepackage{textcomp}
% other useful packages
\usepackage{overpic}
\usepackage[dvipsnames]{xcolor}
\usepackage{hyperref}
\usepackage{pdfpages}
\usepackage{ifoddpage}
\usepackage{textpos}
% for printing used files, as a debug option
\usepackage{xparse,longtable,array}
%\usepackage{xparse,l3regex,longtable,array}
\usepackage{expl3}
\expandafter\def\csname ver@l3regex.sty\endcsname{} 


   \ifxetex
		\usepackage[no-math]{fontspec}
%	 	\setmainfont{TeX Gyre Termes}
		\setmainfont{Adobe Garamond Pro} 
	   	\setsansfont{Frutiger LT Std} %\setsansfont{TeX Gyre Adventor} %	   	\setsansfont{TeX Gyre Heros}
    	\setmonofont{TeX Gyre Cursor}
	\else
		\ClassWarning{cslthse-msc}{Use XeTeX for better fonts!}		
		\usepackage[T1]{fontenc}

%		\usepackage{qtxmath}	
%		\usepackage{tgtermes}
%		\usepackage{tgheros}
%		\usepackage{tgcursor}
%%% change
		\usepackage{lmodern}%\usepackage{tgcursor}
		\usepackage{qtxmath}	
%        \usepackage{tgschola}%\usepackage{tgadventor}%\usepackage{tgheros}
%        \global\let\bfseries\sbweight
        \usepackage{CormorantGaramond}%\usepackage{tgtermes}
%        %for CormorantGaramond
        \let\oldnormalfont\normalfont
        \def\normalfont{\oldnormalfont\mdseries}
%        %endfor CormorantGaramond
%%% end change


%		\InputIfFileExists{garamond.sty}{\AtBeginDocument{\garamond}}{%
%			\ClassError{cslthse-msc}{Missing garamond.sty. Unpack the zip from http://gael-varoquaux.info/computers/garamond/index.html in your source directory.}{}}
	
			
	\fi
\makeatother

% LTH color definitions - could take them out in the class
\definecolor{LTHblue}{RGB}{0,0,128}
\definecolor{LTHbronze}{RGB}{156,97,20}

% the following have also their 50% lighter versions: use \color{LTHpink!50}
\definecolor{LTHgreen}{RGB}{173,202,184}
\definecolor{LTHpink}{RGB}{233,196,199}
\definecolor{LTHcream}{RGB}{214,210,196}
\definecolor{LTHcyan}{RGB}{185,211,220}
\definecolor{LTHgrey}{RGB}{191,184,175}

% fancy headers, footers
\RequirePackage{fancyhdr}
\makeatletter
\def\vhrulefill#1{\leavevmode\leaders\hrule\@height#1\hfill \kern\z@}

\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}{}}
\fancyhf{}
\fancyhead[LE]{\footnotesize{\textsc{\leftmark}}}
\fancyhead[RO]{\footnotesize{\textsc{\rightmark}}}
\fancyfoot[LE,RO]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\fancypagestyle{plain}{
  \fancyhead{}
  \renewcommand{\headrulewidth}{0pt}
%  \renewcommand{\footrulewidth}{0pt}
}
\makeatother

% Title page
\RequirePackage{titling}

\makeatletter

% for listing packages
\ExplSyntaxOn
\NewDocumentCommand{\printfilelist}{}
 {
  \cleardoublepage
  \section*{Used ~ packages}
  \setlength{\LTleft}{0pt}
  \pagestyle{empty}
  \footnotesize
  \begin{longtable}{@{} >{\ttfamily}l >{\raggedright}p{.6\textwidth} @{}}
  \nobbz_print_filelist:
  \end{longtable}
 }

\tl_new:N \l_nobbz_filelist_body_tl

\cs_new_protected:Npn \nobbz_print_filelist:
 {
  \clist_map_inline:cn { @filelist }
   {
    \nobbz_print_fileinfo:n { ##1 }
   }
  \tl_use:N \l_nobbz_filelist_body_tl
 }
 
\cs_new_protected:Npn \nobbz_print_fileinfo:n #1
 {
  \regex_match:nnT { \.(sty|cls)\Z } { #1 }
   {
    \nobbz_print_pkginfo:n { #1 }
   }
 }

\cs_new_protected:Npn \nobbz_print_pkginfo:n #1
 {
  \tl_put_right:Nx \l_nobbz_filelist_body_tl
   {
    \exp_not:n { #1 }
    &
    \exp_not:v { ver@#1 }   
    \exp_not:N \tabularnewline
   }
 }
\ExplSyntaxOff
%

\DeclareOption{nofilelist}{\renewcommand{\printfilelist}{}}
\DeclareOption*{\PackageWarning{cslthse-msc}{Unknown ‘\CurrentOption’}}
\ProcessOptions\relax


    % custom commands
	\newcommand{\company}[1]{\def \@company {#1}}
	\newcommand{\supervisor}[1]{\def \@supervisor {#1}}
	\newcommand{\supervisors}[2]{\def \@firstsupervisor {#1}  \def \@secondsupervisor {#2}}
	\newcommand{\examiner}[1]{\def \@examiner {#1}}
	\newcommand{\subtitle}[1]{\def \@subtitle {#1}}
	\newcommand{\svensktitel}[1]{\def \@svensktitel {#1}}

%	\newcommand{\student}[1]{\def \@student {#1} \def \@author {#1}}
%	\newcommand{\student}[2]{\def \@student {#1} \def \@studentemail {#2} \def \@author {#1 \\ {\normalsize \href{mailto:#2}{\texttt{#2}}}}}
	\newcommand{\student}[2]{\def \@student {#1} \def \@studentemail {#2}}
	\newcommand{\students}[4]{\def \@firststudent {#1} \def \@firststudentemail {#2} \def \@secondstudent {#3} \def \@secondstudentemail {#4}}
	
\newcommand{\studentnames}{
\ifx\@secondstudent\empty%
					\@student
\else%
					\@firststudent, \@secondstudent
\fi%
}

% \def \@author {{{#1} \\ {\normalsize \href{mailto:{#2}}{\texttt{{#2}}}}}}}% \and {{#3} \\ {\normalsize \href{mailto:#4}{\texttt{#4}}}}}}  	
%	\newcommand{\students}[2]{\def \@firststudent {#1} \def \@secondstudent {#2} \def \@author {#1 \and #2}}
	\newcommand{\thesisnumber}[1]{\def \@thesisnumber {#1}}
	\newcommand{\thesistype}[2]{\def \@thesistype {#1} \def \@arbete {#2}}
	\newcommand{\titlebox}[2]{\def \@titleY {#1} \def \@titleH {#2}}
	\newcommand{\onelinetitle}{\titlebox{77}{12}}
	\newcommand{\twolinestitle}{\titlebox{74}{15}}% two lines title
	\newcommand{\threelinestitle}{\titlebox{71}{18}}% three lines title
	\newcommand{\fourlinestitle}{\titlebox{66}{23}}% four lines title



	\company{}
	\supervisor{}
	\supervisors{}{}
	\examiner{}
	\subtitle{}
	\svensktitel{}
	\student{}{}
	\students{}{}{}{}	
	\thesistype{Master}{Examensarbete}
	% for the title page
	\pretitle{	
\ifx\@secondstudent\empty%
	\author{\@student \\ {\normalsize \href{mailto:\@studentemail}{\texttt{\@studentemail}}}}%
\else%
	\author{\@firststudent \\ {\normalsize \href{mailto:\@firststudentemail}{\texttt{\@firststudentemail}}} \and%
\@secondstudent \\ {\normalsize \href{mailto:\@secondstudentemail}{\texttt{\@secondstudentemail}}}}%
\fi%
%
% figure out title box size
%\setcounter{titleY}{80 - 3 * \@titlelines}
%cover page
\begin{textblock*}{\paperwidth}(-3.75cm,-15.7cm)%
%	\begin{overpic}[scale=2.0]{coverjpg.pdf}
\begin{overpic}[scale=.845]{cover.jpg}
%	\begin{overpic}[scale=1.77]{cover.png}
		\put(15,90){\color{white}\large\bfseries\textsf{\MakeUppercase{\@thesistype}'S THESIS 2021}}
%		\put(15,90){\color{white}\Large\bfseries\textsf{BACHELOR'S THESIS \the\year{}}}
%		\put(15,77){\colorbox{white}{\makebox(53,12){ % 77, 12 for single line title 
%		\put(15,74){\colorbox{white}{\makebox(53,15){ % 74, 15 for double line title, 
		\put(15,\@titleY){\colorbox{white}{\makebox(53,\@titleH){ % 71, 18 for tripple line title, 
			\begin{minipage}{0.72\textwidth}
				\color{LTHbronze}
				\begin{flushleft}
%					{\Huge\bfseries\thetitle\ } \\
					{\fontsize{34}{40}\selectfont\bfseries\thetitle\par}
						\vhrulefill{2pt} \\
					\vspace{10pt}
				    {\Large\bfseries \studentnames}				
				\end{flushleft}
			\end{minipage}
		}}}
		\put(1,9){\color{white}{
			\begin{minipage}{0.5\textwidth}
			\begin{flushright}
				{\textsf{\textsc{ISSN 1650-2884}}}\\
				\vspace{5pt}
				{\textsf{\textsc{\@thesisnumber}}} \\ % Define thesis number in the main report file using the number from Birger Swahn - tack Niklas Lundstroem
				\vspace{10pt}
				{\textsf{\MakeUppercase{Department of Computer Science}}}\\
				\vspace{5pt}
				{\textsf{\MakeUppercase{LTH $\mid$ Lund University}}}\\
			\end{flushright}
			\end{minipage}		
		}}
		% bottom white margin...
		\put(0,2){\colorbox{white}{\makebox(68,1){}}}
		% logo
%		\put(32.5,-23.8){\includegraphics[scale=1.2]{LUlogoNEG.png}}		
  		\put(34,-28){\includegraphics[scale=1.2]{LUlogoRGB.png}}		
    \end{overpic}
\end{textblock*}

\clearpage
%\begingroup
  \pagestyle{empty}
  \null
  \newpage
%\endgroup

% now the archive page
\center{\Huge \MakeUppercase{\@arbete}}\\
\center{\huge Datavetenskap}
\vspace{150pt}
\center{\huge \@thesisnumber}
\vspace{50pt}
\center{\huge\bfseries\thetitle\par}
\vspace{20pt}
\center{\huge \@svensktitel\par}
\vspace{60pt}
\center{\Large\bfseries \studentnames}

\clearpage

\pagestyle{empty}
\null
\newpage
\thispagestyle{empty}	
\setcounter{page}{1}
	
	\noindent\rule{\linewidth}{1pt}\begin{center}\Huge\bfseries
	
}
	\posttitle{\par \vskip 0.5em \ifx \@subtitle \empty \vskip 1em \else {\Large(\@subtitle)} \fi \end{center}\noindent\rule{\linewidth}{1pt}\vskip 0.5em}
	\predate{\vskip 5em \begin{center}\Large}
	\postdate{\par\vfill {\large {\@thesistype}'s thesis work carried out at 
		\ifx\@company\empty
		 	\ClassWarning{cslthse-msc}{Host company missing. Use \protect\company{name}. Defaulting to CS.}
		 	\\ the Department of Computer Science, Lund University.
		\else \@company. \fi }
		  
			\vskip 1em 
			{\normalsize
			\ifx\@supervisor\empty 
			   %\ClassWarning{cslthse-msc}{Empty supervisor.}
			   \ifx\@firstsupervisor\empty
				  \ClassError{cslthse-msc}{Missing supervisor. Use either \protect\supervisor{name,email} or \protect\supervisors{name1,email1}{name2,email2}}{You must specify your thesis supervisor(s).}
				\else Supervisors: \@firstsupervisor ~\\ \@secondsupervisor \fi
			\else Supervisor: \@supervisor \fi
			
			Examiner: 
			\ifx\@examiner \empty
				\ClassError{cslthse-msc}{Missing examiner. Use \protect\examiner{name,email}}{You must specify your thesis examiner.}
			\else	 \@examiner \fi } \end{center}}

\makeatother


% Sections and captions
\RequirePackage{titlesec}
\makeatletter
	\titleformat{\chapter}[display]{\normalfont\huge\bfseries}
	{\chaptertitlename\ \thechapter}{10pt}{\Huge}[\vspace{2cm}\hrulefill]
	%\titleformat{\section}[hang]{\LARGE\bfseries\sffamily}{\thesection}{}
	\titleformat*{\section}{\LARGE\bfseries\sffamily}
	\titleformat*{\subsection}{\Large\bfseries\sffamily}
	\titleformat*{\subsubsection}{\large\bfseries\sffamily}
	\titleformat*{\paragraph}{\large\bfseries\sffamily}
	\titleformat*{\subparagraph}{\large\bfseries\sffamily}
\makeatother


% captions for figures and tables
\RequirePackage{caption}
\makeatletter
	\renewcommand{\captionlabelfont}{\bfseries}
	\setlength{\captionmargin}{2cm}
\makeatother

% takes care of the abstract and aknowledgements
\RequirePackage{abstract}

\makeatletter

\newcommand{\keywords}[1]{\def \@keywords {#1}}
\newcommand{\theabstract}[1]{\def \@theabstract {#1}}
\newcommand{\acknowledgements}[1]{\def \@acknowledgements {#1}}

\keywords{}
\theabstract{}
\acknowledgements{}

% indent the abstract properly
\setlength{\absrightindent}{1cm}
\setlength{\absleftindent}{1cm}

% makes everything in the begining of the document
\newcommand{\makefrontmatter}{
	\maketitle

	% make the abstract and keywords
	\ifx \@theabstract \@empty
		\ClassWarning{cslthse-msc}{The abstract is missing. Use \protect\theabstract{text}.}
	\else
		% empty page		
		\pagestyle{empty}
		% make sure the abstract is on a right-hand side
		\cleardoublepage
		\begin{abstract}
		\noindent
		\@theabstract		
		\ifx \@keywords \@empty
			\ClassWarning{cslthse-msc}{Keywords are missing. Use \protect\keywords{words}.}
		\else
			\vspace{2cm}\par\noindent {\small{\bf Keywords\/}: \@keywords}
		\fi
		\end{abstract}
	\fi
	
	
	\pagestyle{fancy}

	% add acknowledgement if any
	\ifx \@acknowledgements \@empty
			\relax
	\else
			\chapter*{Acknowledgements}
			\@acknowledgements
	\fi

        % use a group to ensure parskip is reset after
        \begingroup
            \setlength{\parskip}{0.5em}
            \tableofcontents
        \endgroup
}
\makeatother


% make the bibliography
\makeatletter
\newcommand{\makebibliography}[1]{
	\cleardoublepage
	\phantomsection
	\addcontentsline{toc}{chapter}{\bibname}
	\bibliographystyle{plain}
	\bibliography{#1}
}%

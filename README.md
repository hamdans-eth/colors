# Rutgers University Grounded Semantics Toolkit (RUGSTK), Public Version 1.0

Preface: scipy stack is assumed to be installed (matplotlib,scipy,numpy,etc)

Currently in progress, this is the framework the grounded semantics work of McMahan,Stone, et al.

As of right now, this is accompanying:
B. McMahan and M. Stone 
A Bayesian Model of Grounded Color Semantics
Transactions of the Association for Computational Linguistics, 2015 

It also has the work from:
@article{meomcmahanstone:color,
  author = "T. Meo and B. McMahan and M. Stone",
  title = "Generating and Resolving Vague Color Reference",
  year = 2014,
  journal = "Proceedings of the 18th Workshop Semantics and Pragmatics of Dialogue (SemDial)"
}

Proper bibtex to be inserted. 

To use: put the folder where the parent folder is on the python sys path. (check with 'import sys; print sys.path' 
Or, add it to the sys path ('import sys; sys.path.append(some_directory)')

Note: things get weird when rugstk_v1 (or whatever you name it to) is on the sys path itself. It needs to be a child directory of something on the path. 

Example.py provides example functionality of this package. 



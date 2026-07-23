"""
Act 1 — exact reproduction of the paper's original MATLAB code
(~/matlab_fractalbasin, given by Dr. Fish). Code in this package matches
the original source as closely as possible, including physics choices
this project's earlier (Act 2/3) Lorenz-first pipeline does NOT share --
see talk/notes/matlab_source_audit.md for the full file-by-file comparison
this package is built from.

This is a deliberately separate module from pythongpu.oscillators and
pythongpu.pipeline: those were built around Lorenz (chosen independently,
as a non-brain chaotic system for methodology development -- not a
misreading of the original code) and the streaming-surrogate VPS, and nothing
here should change their behavior. "Exact reproduction" and "the surrogate
approach that came after" are different goals with different code, on
purpose.
"""

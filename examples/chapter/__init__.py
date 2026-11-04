"""Track 1 chapter teaching material (Phase E.2b).

Reader exercises live in ``examples.chapter.sNN_slug`` packages. Each has a
generic ``exercise`` (reader starter), ``solution`` (instructor reference),
and ``test_sNN_slug`` (offline pytest). Shared helpers -- the ``call_llm``
seam, fixture loaders, scripted-response mocks -- live in
``examples.chapter.toolkit``.

Nothing here imports ``slr_pipeline``; the code is self-contained, simplified
teaching code, not a wrapper around the production pipeline.
"""

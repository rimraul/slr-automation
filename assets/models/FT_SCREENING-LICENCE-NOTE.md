# FT_SCREENING licence note

track2_pipeline_readiness_plan_2026-08-30.md §5.2 item 5a: a human-readable
pointer alongside the machine-checked `track2/ft-screening-wheel.lock`, not
a second source of truth -- the lock file's own fields are authoritative;
this note restates them for a reader who has not opened the lock file.

- **Licence commit**: `77817733d4a66c0c810fe2ee12974522aa45ad1f`
  ("License FT_SCREENING under MIT") on FT_SCREENING's own repository.
  Parent: `9604dc7` -- a licence-only commit (one file, `LICENSE`, 21
  lines added, no code hunk).
- **Licence**: MIT, copyright (c) 2026 Rimantas Rauleckas. FT_SCREENING's
  wheel carries this text at its own
  `ft_screening-0.1.0.dev0.dist-info/licenses/LICENSE`
  (`test_ft_screening_wheel_has_mit_licence`).
- **Code parent**: `9604dc7` -- the licence commit changes no source file,
  so the wheel's actual code is identical to `9604dc7`'s.

## Publication history

`77817733d4a66c0c810fe2ee12974522aa45ad1f` was pushed to FT_SCREENING's
GitHub remote (`origin/main`, fast-forward from `9604dc7`) on 2026-09-02,
under the Track 2 step-5 publication packet's authorized-push decision.
The wheel this note accompanies was built from that same commit, locally,
before the push (bit-for-bit reproducible either way, `test_ft_screening_
wheel_reproducible`). This wheel's own upload to the `track2-assets-v1`
release remains a separate, still-pending step (§5.6-C1) -- `gh` is not
yet installed/invoked; see the Track 2 step-5 publication packet for the
exact pending command.

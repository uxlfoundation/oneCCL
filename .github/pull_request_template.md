
> [!WARNING]
> On PR open (including draft PRs), GitHub Actions overwrites the PR title/body with the first commit message (subject/body). Text entered in the PR form is not preserved.
---

## Author Checklist
* [ ] Provide PR Description \
      Particularly focus on why, not what. Reference background, issues, test failures etc.
* [ ] Reference appropriate issues in PR description (with "Fixes" or "See" as appropriate)
* [ ] Commits follow good practice \
      Commits are self-contained and do not do two things at once \
      Commit message references appropriate issues (with "Fixes" or "See" as appropriate) \
      Commit message is of the form: `module: short description` and follows [good practice](https://chris.beams.io/posts/git-commit/) \
      Commit message explains what's in the commit.
* [ ] Add test for new functionality, if needed
* [ ] Tests have passed

## Available test configs

Comment structure **test:\<conf>**

* justbuild|scan
* func
* pr|example|reg_tests
* pr/debug
* example/debug
* reg_tests/debug
* horovod|pytorch|torch_ccl
* kw|abi|format
* mpich_ats|mpich_pvc
* all

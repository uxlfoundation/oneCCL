# Contributing guidelines

We welcome community contributions to oneCCL. You can:

- Submit your changes directly with a [pull request](https://github.com/oneapi-src/oneCCL/pulls).
- Log a bug or feedback with an [issue](https://github.com/oneapi-src/oneCCL/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [isssues](#issues) before you proceed.

## Issues

Use [GitHub issues]((https://github.com/oneapi-src/oneCCL/issues)) to:
- report an issue
- provide feedback
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull requests

Before you submit a pull request, make sure that:

- You follow our [code contribution guidelines](#code-contribution-guidelines) and our [coding style](#coding-style), as well as [commit message guidelines](#commit-message-guidelines).
- You provided the [requested details](#rfc-pull-requests) for new primitives or extended the existing [functional tests](#functional-tests) when fixing an issue.

**Note**: This project follows the
[GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow). To get started with pull requests, see [GitHub howto](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

### Commit and Pull request message guidelines

#### Introduction

The intention of the strict rules for the content and structure of the commit message is to make project history more readable.
Both Conventional Commits [specification](https://www.conventionalcommits.org/en/v1.0.0/) and Angular's [Commit Message Guidelines](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#commit) inspired the rules outlined below

#### Message Format for Commits and Pull Requests

Except reverting Pull Request (PR), the first commit of a PR and the PR message must follow a specific format, consisting of a **header**, a **body**, while the PR message also includes a **footer**.
The main reason the first commit also follows this rule is that its title and message are automatically copied to become the PR message, and eventually become the content of the final commit message when the PR is squashed and merged.
The header has a special format that includes a **type**, a **scope** and a **subject**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **type** is mandatory and the **scope** of the header is optional. Both **type** and **scope** should always be **lowercase**. Multiple values of **type** and **scope** are not allowed.

**subject** summarizes the purpose in a single sentence starting with the present verb form without a period.

Allowed values for the **scope** are basic component or feature to improve, typically categorized by collective, such as `feat(allgather)`.
Allowed values for the **type** are
* feat: A new feature for the user
* fix:  Code changes to fix a bug, not changing or providing any new feature
* perf: Code changes for the performance improvement
* refactor: Code changes to simplfiy the maintenence and/or to reduce complexity, such as splitting single function into multiple ones
* test: Code changes for the test directory only
* build: Changes for the build system or external dependencies, such as CMakeList.txt or build_occl wrapper
* doc: Changes to the Documentation only
* ci: Changes to the CI configuration files and scripts only, such as github and git scripts
* style: Changes that do not affect the meaning of the code, such as whitespace adjustment or clang-format changes
* chore: maintenance and routing tasks, such as .gitignore and dependency bumps

If you have multiple changes that overlap with the **type** mentioned above, use the higher one from the list. For example, your commit provides new feature but changes are in test and improve performance, you should use feat for the commit. Similarly, if you have a PR that has multiple commits for perf, refactor, test, and doc, **type** for the PR should be perf.

Provide a couple sentences of human-readable content in the **body** of the message. Focus on the purpose of the changes, and describe it with present verb form. When **type** is set to **feat**, It is required to describe how the new feature is validated with the existing and/or new tests.

Metadata associated with commit should be included in the **footer** part, which is required for the Pull Request description and its first commit, and optional for any subsequent commits in the PR. Currently, it is expected to contain certificate of origin ( _Signed-Off-By:_ ) and tracker reference ( _Resolves:_ / _Related-To:_ ). Each feature needs to have a design document that explain the full mechanism potentially linked with _Related-To:_ and _Resolves:_.

Note: usage of tracker reference with the _Resolves:_ or _Related-To:_ notation is mandatory for **feat**, **fix**, **perf**, **refactor**, and **test** PRs, and optional for all other values.

As previously mentioned, PRs to revert a prior commit is a special case - use the following commit message format:
```
Revert "<subject of commit being reverted>"

This reverts commit <ID of commit being reverted>
```


### RFC pull requests

It is strongly advised to open an RFC (request for comments) pull request when contributing new
primitives. Please provide the following details:

* The definition of the operation as a oneCCL primitive. It should include an interface and semantics. We welcome sketches for the interface, but the semantics should be fairly well-defined.

* A use case, including a model and parallelism scenario.

### Code contribution guidelines

The code must be:

* *Tested*: oneCCL uses `gtests` for lightweight functional testing.

* *Documented*: oneCCL uses `Doxygen` for inline comments in public header
  files that are used to build the API reference and  `reStructuredText` for the Developer Guide. See [oneCCL documentation](https://oneapi-src.github.io/oneCCL/) for reference.

* *Portable*: oneCCL supports CPU and GPU
  architectures as well as different compilers and run-times. The new code should be complaint
  with the [System Requirements](README.md#prerequisites).

### Coding style

Please refer [Coding Guideline](GUIDELINES.md) for more details.

### Functional tests

How to run functional testing:

1. [Build and install oneCCL](README.md#Installation)
2. Make sure you are located in `<oneCCL directory>/<build directory>`
3. Source oneCCL: `source <oneCCL install directory>/env/setvars.sh`
4. Enter the test directory: `cd tests/functional`
5. Run tests: `ctest -VV -C default`

The results of the tests, including the pass rate, should be printed on the screen.

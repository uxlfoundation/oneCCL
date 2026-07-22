# Introduction

This document defines the organizational structure related to oneCCL
development, defines the roles in the oneCCL project and includes
the current list of oneCCL project members.

# Roles and responsibilities

The oneCCL project defines four main roles:

- [Contributor](#contributor)
- [Core Team Member](#core-team-member)
- [Code Owner](#code-owner)
- [Maintainer](#maintainer)

The project is developed by the Intel Corporation core team. While we
are open to external contributors, the roles of code owners and maintainers
are limited to the core team.

The limitations are related to technical setup of our CI system,
quality assurance procedures and the code release process.

## Contributor

A Contributor invests time and resources to improve oneCCL. Anyone can become
a Contributor by bringing value in one of the following ways:

- Test pull requests and releases and report bugs.
- Contribute code, including:
    - Bug fixes.
    - Features.
    - Performance optimizations.
- Submit a GitHub issue, including:
    - Bug report.
    - Feature requests.
    - Documentation requests.

Responsibilities:

- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Follow the project [contributing guidelines](CONTRIBUTING.md).

Contributors open to long-term cooperation are encouraged
to contact [oneccl.maintainers@intel.com](mailto:oneccl.maintainers@intel.com).

## Core Team Member

A Core Team Member takes care of regular project activities, such as:

- Research.
- Development.
- Bug fixing.
- Code maintenance.

This role is reserved for employees of Intel Corporation.

Responsibilities:

- All responsibilities of [Contributor](#contributor).
- Answer questions from community members.
- Submit review and/or test pull requests.
- Test releases and report bugs when required.
- Contribute source code, including:
    - Bug fixes.
    - Features.
    - Performance optimizations.
- Contribute to documentation.
- Submit internally-tracked issues.
- Follow work agreements with Intel Corporation.
- Follow execution commitment.

Privileges:

- Can approve pull requests.

## Code Owner

A Code Owner supervises work done in their area of responsibility.

This role is reserved for employees of Intel Corporation.

Responsibilities:

- All responsibilities of [Core Team Member](#core-team-member)
- Responds to and answers questions from the community relating to their
                                                        area of ownership.
- Continuously track their area of ownership:
    - Observe activity and code changes in their area of expertise.
    - Report related issues, including the need for additional:
        - Documentation.
        - Refactoring.
        - Optimization.
- Provide guidance about their area of expertise.

Privileges:

- Code owner's approval is required for merging pull requests.
- Can merge approved pull requests.


## Maintainer

Maintainers are responsible for overseeing the entire project. They can:

- Act as project-wide Code Owners.
- Make strategic decisions.

This role is reserved for employees of Intel Corporation.

Responsibilities:

- All responsibilities of [Code Owner](#code-owner)
- Co-own with Code Owners and other Maintainers the project.
- Determine strategy and policy for the entire project.
- Support and guide Contributors, Core Team members and Code Owners.
- Resolve conflicting opinions between Code Owners and/or Contributors.


Privileges:

- Can represent the project in public as a Maintainer.
- Can approve pull requests as Code Owner for the whole project.
- Can merge pull requests.
    - Can override requirements and blocking reviews, if necessary.


# Project members

## Core Team

@oneapi-src/oneccl-core-team

## Code Owners

[CODEOWNERS](.github/CODEOWNERS)

## Maintainers

@oneapi-src/oneccl-maintainers

## Support functions

### Documentation

@oneapi-src/oneccl-doc-team

<!--
SPDX-FileCopyrightText: 2010-2020 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Contributing to AdePT

Contributions to the AdePT project are very welcome and feedback on the documentation is greatly appreciated.

Please also see our [code of conduct](CODE_OF_CONDUCT.md).

## Mailing lists

**FIXME: We need to set these up**

## Bug reports and feature requests

To report an issue and before starting work, please create an issue in the [Github issue tracker](https://github.com/apt-sim/AdePT/issues). A comprehensive explanation will help the development team to respond in a timely manner. Verbalizing the issue before starting work allows the other contributors to chime in and avoids disagreements how to progress.

- The title should summarize the issue
- Describe the issue in as much detail as possible in the comment

Github does not allow editing labels, assignees or setting milestone to non-members of a project with at least "Triage" permission. These will have to be set by members with Triage permission after an issue/PR is created.
Guidelines regarding labels, assignees and milestone therefore only concern members of apt-sim with the necessary rights and can be ignored by others.

- Assign to yourself or leave empty
- Choose labels as appropriate
  - type of issue
  - which component is affected
  - urgency
  - fix versions
- bug reports
  - mention affected version(s)
  - issue type: "Bug"
  - a detailed description of the bug including a recipe on how to reproduce it and any hints which may help diagnosing the problem
- feature requests
  - issue type: "Improvement" or "New Feature"
  - a detailed description of the feature request including possible use cases and benefits for other users

## Make a contribution

Anyone is welcome to contribute to AdePT. Below is a short description how to contribute. If you have any questions, feel free to ask us **FIXME add mailing list** for help or guidance.

Please always fork the AdePT repository and create branches only in your own fork. Once you want to share your work, create a Pull Request (PR) (for gitlab users: equivalent to merge request) to the master branch of the upstream AdePT repository. If it is not yet ready to be merged in, please create a draft pull request (by clicking on the small arrow on the green "create pull request" button) to mark it *work in progress*. Once you want your branch to be merged in, request a review. Once a draft merge request is reviewed, it can be merged in.

To get started with git, please refer to the [short introduction](http://git-scm.com/docs/gittutorial) as well as the [full git documentation](https://git-scm.com/doc). Tutorials as well as explanations of concepts and workflows with git can also be found on [Atlassian](http://www.atlassian.com/git/).

### Checklist for pull requests

- Your branch has been rebased on the target branch and can be integrated through a fast-forward merge.
- A detailed description of the pull request is provided.
- The issue the PR closes is linked.
- All CI jobs pass.
- All newly introduced functions and classes have been documented properly with doxygen.
- Unit tests are provided for new functionalities.
- For bugfixes: a test case has been added to avoid the re-appearance of this bug in the future.
- All added cmake options were added to 'cmake/PrintOptions.cmake'.
- All changes made to the `base/inc/CopCore` subproject should be in separate commits, with no commits containing a mix of files from that project and the main AdePT code.

### Workflow recommendations

In the following a few recommendations are outlined which should help you to get familiar with development process in the AdePT project.

1. **Each development in its own branch of your private fork!**
Write access for developers has been disabled for developers on AdePT-project. Therefore, always start by creating your own fork and creating branches there. You should start a new branch for every development and all work which is logically/conceptually linked should happen in one branch. Try to keep your branches short. This helps immensely to understand the git history if you need to look at it in the future and helps reviewers of your code.
If projects are complex (e.g. large code refactoring or complex new features), you may want to use _sub_-branches from the main development branch.

1. **Never, ever directly work on any "official" branch!**
Though not strictly necessary and in the end it is up to you, it is strongly recommended that you never commit directly on a branch which tracks an "official" branch. As all branches are equal in git, the definition of "official" branch is quite subjective. In the AdePT project you should not work directly on branches which are **protected** in the repository. Usually, these are the _master_ and _release/X.Y.Z_ branches. The benefit of this strategy is that you will never have problems to update your fork. Any git merge in your local repository on such an "official" branch will always be a fast-forward merge.

1. **Use atomic commits!**
Similarly to the concept of branches, each commit should reflect a self-contained change. Try to avoid overly large commits (bad examples are for instance mixing logical change with code cleanup and typo fixes).
In particular, put changes to the `base/inc/CopCore` subproject in separate commits from any other changes.

1. **Write good commit messages!**
Well-written commit messages are key to understand your changes. There are many guidelines available on how to write proper commit logs (e.g. [here](http://alistapart.com/article/the-art-of-the-commit), [here](http://chris.beams.io/posts/git-commit/), or [here](https://wiki.openstack.org/wiki/GitCommitMessages#Information_in_commit_messages)). As a short summary:
    - Structure your commit messages into short title (max 50 characters) and longer description (max width 72 characters)!
      This is best achieved by avoiding the `commit -m` option. Instead write the commit message in an editor/git tool/IDE...
    - Describe why you did the change (git diff already tells you what has changed)!
    - Mention any side effects/implications/consequences!

1. **Prefer git pull --rebase!**
If you work with a colleague on a new development, you may want to include his latest changes. This is usually done by calling `git pull` which will synchronise your local working copy with the remote repository (which may have been updated by your colleague). By default, this action creates a merge commit if you have local commits which were not yet published to the remote repository. These merge commits are considered to contribute little information to the development process of the feature and they clutter the history (read more e.g.  [here](https://developer.atlassian.com/blog/2016/04/stop-foxtrots-now/) or [here](http://victorlin.me/posts/2013/09/30/keep-a-readable-git-history)). This problem can be avoided by using `git pull --rebase` which replays your local (un-pushed) commits on the tip of the remote branch. You can make this the default behaviour by running `git config pull.rebase true`. More about merging vs rebasing can be found [here](https://www.atlassian.com/git/tutorials/merging-vs-rebasing/).

1. **Update the documentation!**
Make sure that the documentation is still valid after your changes. Perform updates where needed and ensure integrity between the code and its documentation.

### Coding style and guidelines

#### C++

The AdePT project uses [clang-format](http://clang.llvm.org/docs/ClangFormat.html) for formatting its C++ source code. A `.clang-format` configuration file comes with the project **FIXME** and should be used to automatically format the code. Developers can use the provided Docker image to format their project or install clang-format locally. Developers should be aware that clang-format will behave differently for different versions, so installing the same clang version as used in the CI **FIXME** is recommended. There are several instructions available on how to integrate clang-format with your favourite IDE (e.g. [eclipse](https://marketplace.eclipse.org/content/cppstyle), [Xcode](https://github.com/travisjeffery/ClangFormat-Xcode), [emacs](http://clang.llvm.org/docs/ClangFormat.html#emacs-integration)). The AdePT CI system will automatically apply code reformatting using the provided clang-format configuration once pull requests are opened. However, developers are encouraged to use this code formatter also locally to reduce conflicts due to formatting issues.

In addition, some conventions are used in AdePT code, details can be found [here](https://AdePT.readthedocs.io/en/latest/codeguide.html). For Doxygen documentation, please follow these recommendations:

- Put all documentation in the header files.
- Use `///` as block comment (instead of `/* ... */`).
- Doxygen documentation goes in front of the documented entity (class, function, (member) variable).
- Use the `@<cmd>` notation.
- Document all (template) parameters using \@(t)param and explain the return value for non-void functions. Mention important conditions which may affect the return value.
- Use `@remark` to specify pre-conditions.
- Use `@note` to provide additional information.
- Link other related entities (e.g. functions) using `@sa`.

#### Python

Any Python code should be formatted with `black` and should be clean when run through the `flake8` Python linter. There is a `.flake8` file in the repository that will set the configuration to be compatible with `black`.

#### Markdown

This is **TBD**!

Markdown files should conform to the `markdownlint` specifications from [David Anson](https://github.com/DavidAnson/markdownlint).

### Example: Make a bugfix while working on a feature

During the development of a new feature you discover a bug which needs to be fixed. In order to not mix bugfix and feature development, the bugfix should happen in a different branch. The recommended procedure for handling this situation is the following:

1. Get into a clean state of your working directory on your feature branch (either by committing open changes or by stashing them).
1. Checkout the branch the bugfix should be merged into (either _master_ or _release/X.Y.Z_) and get the most recent version.
1. Create a new branch for the bugfix.
1. Fix the bug, write a test, update documentation etc.
1. Open a pull request for the bug fix.
1. Switch back to your feature branch.
1. Merge your local bugfix branch into the feature branch. Continue your feature development.
1. Eventually, the bugfix will be merged into _master_. Then, you can rebase your feature branch on master which will remove all duplicate commits related to the bugfix.

### Example: Backporting a feature or bugfix

Suppose you have a bugfix or feature branch that is eventually going to be merged in `master`. You might want to have the feature/bugfix available in a patch (say `0.25.1`) tag. To to that, find the corresponding release branch, for this example that would be `release/v0.25.X`. You must create a dedicated branch that **only** contains the commits that relate to your feature/bugfix, otherwise the PR will contain all other commits that were merged into master since the release was branched off. With that branch, open a PR to that branch, and make it clear that it is a backport, also linking to a potential equivalent PR that targets `master`.

### Tips for users migrating from Gitlab

- The most obvious difference first: What is called Merge Request (MR) in GitLab is called Pull Request (PR) in Github.
- Once your PR is ready to be merged, request a review by the users in the [reviewers team](https://github.com/orgs/AdePT-project/teams/reviewers)
- As AdePT started enforcing using your own fork with the switch to Github, developers no longer have write access to the upstream repository.
- The CI will fail if a PR does not yet have the required approvals.

## Review other contributions

AdePT requires that every pull request receives at least one review by a member of the reviewers team before being merged but anyone is welcome to contribute by commenting on code changes. You can help reviewing proposed contributions by going to [the "pull requests" section of the AdePT (core) Github repository](https://github.com/apt-sim/AdePT/pulls).

As some of the guidelines recommended here require rights granted to the reviewers team, this guide specifically addresses the people in this team. The present contribution guide should serve as a good indication of what we expect from code submissions.

### Configuring your own PRs and PRs by people with read rights

- Check if the "request review" label is set, if it is a draft PR or if the title starts with WIP. The latter two are equivalent.
- Check if the "triage" label is set and configure labels, assignees and milestone for those PR
  - Needs at least label "Bug", "Improvement", "Infrastructure" or "New Feature"

### Approving a pull request

- Does its title and description reflect its contents?
- Do the automated continuous integration tests pass without problems?
- Have all the comments raised by previous reviewers been addressed?

If you are confident that a pull request is ready for integration, please make it known by clicking the "Approve pull request" button of the GitLab interface. This notifies other members of the AdePT team of your decision, and marks the pull request as ready to be merged.

### Merging a pull request

If you have been granted write access on the AdePT repository, you can merge a pull request into the AdePT master branch after it has been approved.

Github may warn you that a "Fast-forward merge is not possible". This warning means that the pull request has fallen behind the current AdePT master branch, and should be updated through a rebase. Please notify the pull request author in order to make sure that the latest master changes do not affect the pull request, and to have it updated as appropriate.

For a PR that is behind master, a button "Update branch" may appear. This should NOT be used as it merges instead of rebasing, which is not our workflow.

## Acknowledgement

This contribution guide was inspired by the guide from the [Acts project](https://github.com/acts-project/acts). Thanks to those authors!

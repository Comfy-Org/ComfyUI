# Contributing to ComfyUI

Welcome, and thank you for your interest in contributing to ComfyUI!

There are several ways in which you can contribute, beyond writing code. The goal of this document is to provide a high-level overview of how you can get involved.

## Asking Questions

Have a question? Instead of opening an issue, please ask on [Discord](https://comfy.org/discord) or [Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) channels. Our team and the community will help you.

## Providing Feedback

Your comments and feedback are welcome, and the development team is available via a handful of different channels.

See the `#bug-report`, `#feature-request` and `#feedback` channels on Discord.

## Reporting Issues

Have you identified a reproducible problem in ComfyUI? Do you have a feature request? We want to hear about it! Here's how you can report your issue as effectively as possible.


### Look For an Existing Issue

Before you create a new issue, please do a search in [open issues](https://github.com/comfyanonymous/ComfyUI/issues) to see if the issue or feature request has already been filed.

If you find your issue already exists, make relevant comments and add your [reaction](https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments). Use a reaction in place of a "+1" comment:

* 👍 - upvote
* 👎 - downvote

If you cannot find an existing issue that describes your bug or feature, create a new issue. We have an issue template in place to organize new issues.


### Creating Pull Requests

* Please refer to the article on [creating pull requests](https://github.com/comfyanonymous/ComfyUI/wiki/How-to-Contribute-Code) and contributing to this project.


## Release Process

ComfyUI aims for a weekly release cycle targeting Monday, though this shifts regularly
around model releases and large codebase changes. Three repositories are involved.

**[ComfyUI Core](https://github.com/Comfy-Org/ComfyUI)**

* A new major stable version (for example v0.7.0) lands roughly every 2 weeks.
* Since v0.4.0, patch versions carry fixes backported onto the current stable release.
* Minor versions are used for releases off the `master` branch. Patch versions may also
  be used on `master` where a backport would not make sense.
* Commits outside the stable release tags can be very unstable and may break many custom
  nodes.
* Serves as the foundation for the desktop release.

**[Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop)**

* Builds a new release using the latest stable core version.

**[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**

* Frontend updates are merged into the core repository every 2+ weeks.
* Features are frozen for the upcoming core release, and development continues for the
  next cycle.


## Thank You

Your contributions to open source, large or small, make great projects like this possible. Thank you for taking the time to contribute.

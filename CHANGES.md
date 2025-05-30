# ASTERIA Change Log

## 1.0.0 (2020-07-01)

* Added SimulationHandler to manage simulation setup/teardown ([PR #52](https://github.com/icecube/ASTERIA/pull/52)).
* Added build and test requirements for CircleCI and PyPI ([PR #10](https://github.com/icecube/ASTERIA/pull/10)).
* Implemented I/O module for saving and loading simulations ([PR #8](https://github.com/icecube/ASTERIA/pull/8), [PR #38](https://github.com/icecube/ASTERIA/pull/38)).
* Added computation of neutrino oscillations ([PR #6](https://github.com/icecube/ASTERIA/pull/6), [PR #11](https://github.com/icecube/ASTERIA/pull/11), [PR #16](https://github.com/icecube/ASTERIA/pull/16), [PR #19](https://github.com/icecube/ASTERIA/pull/19), [PR #30](https://github.com/icecube/ASTERIA/pull/30), [PR #35](https://github.com/icecube/ASTERIA/pull/35), [PR #44](https://github.com/icecube/ASTERIA/pull/44)).
* Cleaned up Cherenkov light calculations ([PR #5](https://github.com/icecube/ASTERIA/pull/5)).

## 0.2.0 (2019-02-24)

* Added stellar CDFs in FITS format to generate random progenitor distances ([PR #4](https://github.com/icecube/ASTERIA/pull/4)).
* Fixed non-functional unit tests ([PR #3](https://github.com/icecube/ASTERIA/pull/3)).
* Converted references to ussr/USSR module name to asteria/ASTERIA ([PR #2](https://github.com/icecube/ASTERIA/pull/2)).
* Correcting scaling differences between USSR and ASTERIA, connecting detector response to luminosity ([PR #1](https://github.com/icecube/ASTERIA/pull/1)).
* Added basic unit tests.
* Added geometry and effective volume files.
* Cleaned up flavor enums for CCSN calculation.
* Updated README with installation instructions.

## 0.1.0 (2018-06-02)

Initial release.

* Ported USSR calculations for neutrino interactions.
* Added test notebooks for IBD calculation and cross section evaluations.

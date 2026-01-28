## [1.4.0](https://github.com/ibm/ado/compare/1.3.3..1.4.0) - 2026-01-28
#### Features
- (**core**) allow specifying actuator in csv sample store (#455) - ([bda00ad](https://github.com/ibm/ado/commit/bda00adfa1edf6c86acc27a8d68bac1a0a4d13f5)) - Michael Johnston
- (**ray_tune**) add difference stopper (#412) - ([3d2f46c](https://github.com/ibm/ado/commit/3d2f46ca9eb71464e9fc357620659b68307b3af5)) - Michael Johnston
- (**sfttrainer**) add support for fms-hf-tuning==v3.1.0 (#442) - ([b060d3d](https://github.com/ibm/ado/commit/b060d3d4d861211808131cdcd214501631f1aeb6)) - Vassilis Vassiliadis
- (**vllm_performance**) Add GuideLLM experiments (#459) - ([46ffac8](https://github.com/ibm/ado/commit/46ffac88b47cb4c71c92a480c40028876955103e)) - Christian Pinto
- (**vllm_performance**) ensure namespace is RFC1123-compliant (#358) - ([2bf1072](https://github.com/ibm/ado/commit/2bf107216b66f49050c709f41ea2336ac13730f4)) - Alessandro Pomponio
#### Bug Fixes
- (**core**) add relationships to interrupted nested operations  (#379) - ([e6eacdb](https://github.com/ibm/ado/commit/e6eacdbb1a9f8481181319937a1bb59f32b41b4c)) - Michael Johnston
- (**modules**) use correct type annotations for closures (#447) - ([335903b](https://github.com/ibm/ado/commit/335903be04ef60a38719822005d8d7f310001346)) - Alessandro Pomponio
- (**ordered_pip**) keep fields other than ordered_pip and inject pip_install_options to all phases (#390) - ([f8f9def](https://github.com/ibm/ado/commit/f8f9def1924183eb8d95b9ff7a9cabb16d4f2c99)) - Vassilis Vassiliadis
- (**sfttrainer**) fix the support for fms-hf-tuning==3.1.0 (#448) - ([e9bf2b2](https://github.com/ibm/ado/commit/e9bf2b22e9c337a6a21c2e75bc8f15041bdd33a2)) - Vassilis Vassiliadis
- (**sfttrainer**) the only ado wheel to propagate is ado-sfttrainer (#431) - ([b7583ec](https://github.com/ibm/ado/commit/b7583ecd7247990fde3b1ee8703999293f269948)) - Vassilis Vassiliadis
#### Performance Improvements
- use new numpy RNG apis (#357) - ([1c9c7f3](https://github.com/ibm/ado/commit/1c9c7f3ad0e0277044bdaf9c79b0c49dda1ac25f)) - Alessandro Pomponio
#### Documentation
- (**changelog**) update changelog (#377) - ([7372902](https://github.com/ibm/ado/commit/7372902b2795416baa3601625c1307b075fa0c74)) - Alessandro Pomponio
- (**sfttrainer**) update documentation for HYBRID_SHARD (#389) - ([4364b4e](https://github.com/ibm/ado/commit/4364b4ec8d6cae71bb8e9ac8ca031202cf311965)) - Vassilis Vassiliadis
#### Tests
- (**core**) add coverage report (#391) - ([d7afb9c](https://github.com/ibm/ado/commit/d7afb9cf3468df7fa2e4dcd58becbce66e299121)) - Alessandro Pomponio
#### Build system
- (**deps**) update dependencies (#466) - ([1a12fa6](https://github.com/ibm/ado/commit/1a12fa6524b073effb0ef0ad0f8be81c24af477f)) - Alessandro Pomponio
- (**deps**) update dependencies (#451) - ([e1dc4a1](https://github.com/ibm/ado/commit/e1dc4a19f3e241c8c438c477b3c351c80076cc5a)) - Alessandro Pomponio
- (**deps**) update dependencies (#430) - ([efe881d](https://github.com/ibm/ado/commit/efe881d165f30e011880201bea8ab0f769f36731)) - Alessandro Pomponio
- (**deps**) update dependencies (#361) - ([ef91044](https://github.com/ibm/ado/commit/ef91044994bf8716f17629b470f2072cbe913512)) - Alessandro Pomponio
- (**ruff**) enable ANN linter (#440) - ([a0c8a63](https://github.com/ibm/ado/commit/a0c8a6345d657cc738ded37e8a3c7356ffbd97fe)) - Alessandro Pomponio
- (**vllm_performance**) update dependencies (#439) - ([b5162ec](https://github.com/ibm/ado/commit/b5162ecd021eb6940c5cf85ce9583b4510d6763c)) - Alessandro Pomponio
- (**vllm_performance**) require vllm>=0.12.0 (#426) - ([866c067](https://github.com/ibm/ado/commit/866c067abd35477405960a1503838d5dc77f31a3)) - Alessandro Pomponio
- update pre-commit hooks (#371) - ([b051c50](https://github.com/ibm/ado/commit/b051c50f0a1bebf086f30f3f755c4f90758c0543)) - Alessandro Pomponio
#### Refactoring
- (**core**) mark ADOResource.identifier as Defaultable (#446) - ([93b8e34](https://github.com/ibm/ado/commit/93b8e34571bad8b7ca5ef2dd1d496071206de903)) - Alessandro Pomponio
- (**core**) change validate_model to instance method (#370) - ([2020eb6](https://github.com/ibm/ado/commit/2020eb605b44ecbefb1bb70eb63a65c646f16d4e)) - Alessandro Pomponio
- (**samplestores**) disallow None parameters (#408) - ([2b31405](https://github.com/ibm/ado/commit/2b31405bbbf8372daa8fb20717cb53b4ea232277)) - Alessandro Pomponio
- (**sfttrainer**) rewrite simple string joins as fstrings (#454) - ([d6bf32a](https://github.com/ibm/ado/commit/d6bf32a7146d492e8b6296f6e2d0a1dfdd2439db)) - Alessandro Pomponio
- (**tests**) clean up tox file (#465) - ([994b2f7](https://github.com/ibm/ado/commit/994b2f716ef864430b5ed8cf8533c7cb3cf84e3f)) - Alessandro Pomponio
- (**vllm_performance**) remove upgrade path for parameters (#418) - ([7746924](https://github.com/ibm/ado/commit/77469248c64c6b9cdd4e815c14bbcb589f04976e)) - Alessandro Pomponio
- (**vllm_performance**) avoid invoking bench command with shell=True (#359) - ([0314c02](https://github.com/ibm/ado/commit/0314c026d280199e242ccf545f27ae3082158353)) - Alessandro Pomponio
- remove commented-out code (#453) - ([46a645f](https://github.com/ibm/ado/commit/46a645f57abe9e7bf23e7c483ec8c43cbe044857)) - Alessandro Pomponio
- use Annotated pattern for pydantic models (#443) - ([526af2e](https://github.com/ibm/ado/commit/526af2eb791106ddcde83acdcb563bf8f03d034c)) - Alessandro Pomponio
- use Annotated type hint pattern for Pydantic models (#436) - ([9a83af2](https://github.com/ibm/ado/commit/9a83af245b731fce62a95a2b623fe0154833eec4)) - Alessandro Pomponio
- auto-add type annotations where possible (#393) - ([c7daab5](https://github.com/ibm/ado/commit/c7daab58a22f03d028691e66ac8d74662b3a83d7)) - Alessandro Pomponio
- enable ruff's Bandit linter (S) (#365) - ([5eb333f](https://github.com/ibm/ado/commit/5eb333f7d49609b0ed6a9d5df5f36f677218fb87)) - Alessandro Pomponio
- rewrite PropertyDescriptor.__eq__ (#349) - ([0ce1db6](https://github.com/ibm/ado/commit/0ce1db6fdc4bb0889e5f6bf7b2d236a7d7414ccb)) - Alessandro Pomponio
#### Style
- (**anomalous_series**) add type annotations (#425) - ([31f20ae](https://github.com/ibm/ado/commit/31f20aedab9b26c4655899c75a11f17e7f40215a)) - Alessandro Pomponio
- (**autoconf**) add type annotations (#423) - ([7a3bd38](https://github.com/ibm/ado/commit/7a3bd384e18926549de8de49069877733fa4adb1)) - Alessandro Pomponio
- (**cli**) add type annotations (#400) - ([381e645](https://github.com/ibm/ado/commit/381e6456a9b6d43942ed3d2d75bba2a4dc2ec8f0)) - Alessandro Pomponio
- (**core**) add type annotations (#414) - ([bb9039e](https://github.com/ibm/ado/commit/bb9039e8988f996ea45bab42864f8c887f3f1f76)) - Alessandro Pomponio
- (**example_actuator**) add type annotations (#421) - ([9efc5b4](https://github.com/ibm/ado/commit/9efc5b47419e82a68698d16e1bf9fb05b43d5e5c)) - Alessandro Pomponio
- (**examples**) add type annotations (#434) - ([174d5a9](https://github.com/ibm/ado/commit/174d5a91d8d361e454f43069b223b56d3c75b206)) - Alessandro Pomponio
- (**metastore**) add type annotations (#402) - ([41b57f1](https://github.com/ibm/ado/commit/41b57f151fef80fdee665bd3778e729d4349ce5f)) - Alessandro Pomponio
- (**modules**) add type annotations (#413) - ([4adaaf9](https://github.com/ibm/ado/commit/4adaaf9467b750bb1efeb6ba043cbaa55f3193c7)) - Alessandro Pomponio
- (**ray_tune**) add type annotations (#437) - ([6cde7a1](https://github.com/ibm/ado/commit/6cde7a1a6ed0a665f0ca133ec973e3a9457dd438)) - Michael Johnston
- (**samplestores**) add type annotations (#404) - ([6d2c683](https://github.com/ibm/ado/commit/6d2c6837dcfdfe4867a8688485716a4aceb13496)) - Alessandro Pomponio
- (**schema**) add type annotations (#416) - ([a215ec7](https://github.com/ibm/ado/commit/a215ec7874635f677f983c2f7e3f64ff6c298e85)) - Alessandro Pomponio
- (**sfttrainer**) add type annotations (#420) - ([ca5548f](https://github.com/ibm/ado/commit/ca5548f8407338def3b5d02c4cf4b63a74665d88)) - Alessandro Pomponio
- (**tests**) add type annotations (#432) - ([44bea4d](https://github.com/ibm/ado/commit/44bea4de54b24ab04e9e6ceece52555783631242)) - Alessandro Pomponio
- (**utilities**) add type annotations (#406) - ([c4c6a63](https://github.com/ibm/ado/commit/c4c6a6362c6bac88fa32901a35acf38d20a7d7ce)) - Alessandro Pomponio
- enable PIE and T10 linters (#384) - ([3d2f9f5](https://github.com/ibm/ado/commit/3d2f9f567b78682dddd22e985d600cd622320101)) - Alessandro Pomponio

## [1.3.3](https://github.com/ibm/ado/compare/2fc3b9113577a2e7a9fb7532440b166237c7f6fa..1.3.3) - 2026-01-8
#### Features
- (**cli**) support loading commands via plugins (#344) - ([6e61cff](https://github.com/ibm/ado/commit/6e61cff68d90004617b4d3b8f00ec116bd9e7651)) - Michael Johnston
#### Bug Fixes
- (**cli**) update calculations in show details operation (#325) - ([dfd4884](https://github.com/ibm/ado/commit/dfd4884d98e6ff08d0a5c4eba87e52055a72806c)) - Alessandro Pomponio
- (**core**) update PropertyValue schema for structured decoding  (#350) - ([58b5fd2](https://github.com/ibm/ado/commit/58b5fd20be93d3158220a17f69c543bce1b792c1)) - Michael Johnston
- (**docs**) update changelog link in pyproject (#351) - ([f1094df](https://github.com/ibm/ado/commit/f1094dfd2108ca1009b9d67b6f6403afe137a64d)) - Alessandro Pomponio
- (**docs**) update docs for upgrading actuator configurations (#343) - ([7a5804d](https://github.com/ibm/ado/commit/7a5804df7d7a13acf006e30cacd248cdd7d986e3)) - Alessandro Pomponio
- enable Bugbear linter (#330) - ([f580b34](https://github.com/ibm/ado/commit/f580b34abca11db10651fe47ca51a55902245c97)) - Alessandro Pomponio
#### Performance Improvements
- enable PERF linter (#333) - ([179c2b6](https://github.com/ibm/ado/commit/179c2b6c33d209213535a6625c78a2c699e5b070)) - Alessandro Pomponio
#### Documentation
- (**changelog**) update changelog (#321) - ([66a60f0](https://github.com/ibm/ado/commit/66a60f02d14f39ccbadfaa1c13471f52d86e39e5)) - Alessandro Pomponio
- (**vllm_performance**) fix in_cluster spelling (#326) - ([6b9f639](https://github.com/ibm/ado/commit/6b9f639b127bb9e6cb8fbda0440cd8b38b01bc12)) - Christian Pinto
- (**website**) fix typo in "Target v observed property formats" (#345) - ([6da973a](https://github.com/ibm/ado/commit/6da973a20947c10bc6a69ea9340483962d871c05)) - Daniele Lotito
#### Build system
- (**containers**) support geo and sft image (#341) - ([dd0c0c7](https://github.com/ibm/ado/commit/dd0c0c78ed79c3bec28cdce7e707c9452cceb5fc)) - Alessandro Pomponio
- (**containers**) support building on multiple Python and CUDA versions (#318) - ([4472e1d](https://github.com/ibm/ado/commit/4472e1df58ea3833fb82e89b39803f579d648b06)) - Alessandro Pomponio
- (**core**) add required environments (#323) - ([5590c16](https://github.com/ibm/ado/commit/5590c1663c0c9430968087038294756ab5767be2)) - Alessandro Pomponio
- (**deps**) update dependencies (#353) - ([b6737ef](https://github.com/ibm/ado/commit/b6737ef74c54e7463db0ca9d46bd88de1bb78282)) - Alessandro Pomponio
- (**deps**) update dependencies (#336) - ([2888b7f](https://github.com/ibm/ado/commit/2888b7fb61ad1d736174bb123446518599f30480)) - Alessandro Pomponio
- (**deps**) update dependencies (#331) - ([42c3c0f](https://github.com/ibm/ado/commit/42c3c0f7a657ff87d464ff5a81aab732d103fa25)) - Alessandro Pomponio
- (**deps**) update dependencies (#320) - ([2fc3b91](https://github.com/ibm/ado/commit/2fc3b9113577a2e7a9fb7532440b166237c7f6fa)) - Alessandro Pomponio
#### Refactoring
- (**core**) delay expensive imports (#328) - ([28963fd](https://github.com/ibm/ado/commit/28963fda7b1d51bdad0572d48b6a726856536d29)) - Michael Johnston
- rewrite ProbabilityFunction.__eq__ (#348) - ([6282363](https://github.com/ibm/ado/commit/6282363ea52e244e1d8bd5d9bd7ff7346c0056ce)) - Alessandro Pomponio
#### Miscellaneous Chores
- update gitignore (#342) - ([3e14b3d](https://github.com/ibm/ado/commit/3e14b3d7ffcc2dfad1a71e6cc9df4e6ffe3bf201)) - Alessandro Pomponio

## [1.3.2](https://github.com/ibm/ado/compare/1.3.1..1.3.2) - 2025-12-16
#### Features
- (**core**) handle errors per custom_experiment (#314) - ([67f69cc](https://github.com/ibm/ado/commit/67f69cc0a18ea1d773811481ed5eaa2636d61f33)) - Michael Johnston
- (**custom_experiments**) enforce stricter rules on outputs (#315) - ([8a2e924](https://github.com/ibm/ado/commit/8a2e924ab8148001b8c329339bb63bbb8f5a98e5)) - Michael Johnston
- (**ray_tune**) support multi-objective optimization with optuna (#307) - ([d27f76c](https://github.com/ibm/ado/commit/d27f76cf14c2669062b4caccfbcf18e93e19420d)) - Michael Johnston
- (**vllm_performance**) add support for benchmarking geospatial models (#187) - ([541eaee](https://github.com/ibm/ado/commit/541eaee317dc20faa0283203c250997f69212394)) - Christian Pinto
#### Bug Fixes
- (**ray_tune**) update imports in LHC sampler (#310) - ([63a1484](https://github.com/ibm/ado/commit/63a1484435e4e2b2e42a40f85298ec420642813a)) - Michael Johnston
- (**run_experiment**) print request series with use_markup=False (#319) - ([6e9c078](https://github.com/ibm/ado/commit/6e9c0780f31e70a84a97565d1fa7861eb26b66d5)) - Michael Johnston
- (**vllm_performance**) add missing parameter to execute_random_benchmark and make geospatial experiments beta (#317) - ([05e2713](https://github.com/ibm/ado/commit/05e271372e6ae2048fde3ba04667707a501fe550)) - Christian Pinto
#### Build system
- (**deps**) update dependencies (#311) - ([666defc](https://github.com/ibm/ado/commit/666defc2e32eac8f249d14dfb42c1a38abf296f7)) - Alessandro Pomponio
#### Refactoring
- (**core**) separate cleanup logic from signal handling and fix nested-operation shutdown (#281) - ([1405774](https://github.com/ibm/ado/commit/1405774d1efc1fd3e453d2cfba010105498fcc27)) - Michael Johnston

## [1.3.1](https://github.com/ibm/ado/compare/1.3.0..1.3.1) - 2025-12-10
#### Bug Fixes
- (**cli**) do not use rich's Console.print with dataframes (#297) - ([97c6bea](https://github.com/ibm/ado/commit/97c6beaf1dc66a59f1c9e45f19fada66221df450)) - Alessandro Pomponio
#### Documentation
- (**changelog**) update changelog (#287) - ([6b63d40](https://github.com/ibm/ado/commit/6b63d4045b6e9f3629d92f0757c868da91ba7318)) - Alessandro Pomponio
- (**website**) update instructions to build python wheels for ado and plugins (#301) - ([a62af24](https://github.com/ibm/ado/commit/a62af246b3853d6b4b223689cc646c8c4e74168d)) - Vassilis Vassiliadis
- (**website**) simplify cli examples  (#293) - ([726aec9](https://github.com/ibm/ado/commit/726aec9aceceb7f4f1f02d01c8651a3ee0a08eb4)) - Michael Johnston
#### Build system
- (**autoconf**) pin the required autogluon version (#304) - ([d51f324](https://github.com/ibm/ado/commit/d51f32474c5072b379ecad43d10b5ab4ccad8353)) - Srikumar Venugopal
- (**deps**) update dependencies (#300) - ([a247dfe](https://github.com/ibm/ado/commit/a247dfe946156bcd661d113838bdb1613a2d97e7)) - Alessandro Pomponio
- support Python 3.13 (#291) - ([0ea5cbb](https://github.com/ibm/ado/commit/0ea5cbb32c9e29df78db33aaf88afbd305305cd6)) - Alessandro Pomponio
- update pre-commit hooks (#298) - ([3ff6a6e](https://github.com/ibm/ado/commit/3ff6a6ec0187224991ad72da8842ce4e3517cd3d)) - Alessandro Pomponio
#### Refactoring
- (**cli**) improve sizing of live results table during operations (#299) - ([83716ac](https://github.com/ibm/ado/commit/83716acd1dbac5e9815b06b131dddc85e2c814b1)) - Alessandro Pomponio
- (**run_experiment**) replace prints with console_prints (#289) - ([e728921](https://github.com/ibm/ado/commit/e72892194c3ad1be5f9ccea5406d14d8cf5b028b)) - Alessandro Pomponio
#### Style
- format yaml files with yamlfmt (#286) - ([e2eadfd](https://github.com/ibm/ado/commit/e2eadfdf4caa32a9a493fe8aa415db1bd122d7b8)) - Alessandro Pomponio

## [1.3.0](https://github.com/ibm/ado/compare/6de12d6c25d9ecd9685919b9192e9c0ddc6bbee7..1.3.0) - 2025-12-04
#### Features
- (**autoconf**) introduce autoconf custom experiments (#255) - ([3c1fd87](https://github.com/ibm/ado/commit/3c1fd87ac13d067d31499701031da537b7428cc3)) - Srikumar Venugopal
- (**cli**) support --with in ado create (#262) - ([6de12d6](https://github.com/ibm/ado/commit/6de12d6c25d9ecd9685919b9192e9c0ddc6bbee7)) - Alessandro Pomponio
- (**core**) allow custom_experiments to execute with or without Ray (#263) - ([ea4cab7](https://github.com/ibm/ado/commit/ea4cab720cdf89023c1d176da3a4336f24fb5d98)) - Michael Johnston
- (**sfttrainer**) support granite-3.3-8b (#276) - ([3d1733c](https://github.com/ibm/ado/commit/3d1733c7b429c4f3ba3efe1f2a03a3c1abd500ef)) - Vassilis Vassiliadis
#### Bug Fixes
- (**custom_experiments**) required_properties parameter of decorator was required instead of optional (#278) - ([bec1a19](https://github.com/ibm/ado/commit/bec1a19277204b0d5f80292802ac5eba70261e00)) - Michael Johnston
- (**ordered_pip**) re-create venv if it has been garbage collected (#285) - ([b327f16](https://github.com/ibm/ado/commit/b327f16ee988ed2027b0dac4ac824d732905031d)) - Vassilis Vassiliadis
- (**vllm_performance**) Avoid multiple experiments using the same kubernetes deployment at the same time (#268) - ([34a64af](https://github.com/ibm/ado/commit/34a64aff6ab900486d35c922b2b83431212f714b)) - Christian Pinto
#### Documentation
- (**changelog**) update changelog (#270) - ([c768436](https://github.com/ibm/ado/commit/c7684361b90e6d2e500665339a256007cc6b6ac5)) - Alessandro Pomponio
- (**test**) Add --reinstall flag to uv sync command (#274) - ([40ce0b0](https://github.com/ibm/ado/commit/40ce0b02d6601568a8b4e2125f8eaf02b6c772eb)) - Michael Johnston
- (**website**) more robust custom experiment docs (#284) - ([30f0eb3](https://github.com/ibm/ado/commit/30f0eb3af68d8c46b74dc9a42e3cfccd7cc80274)) - Michael Johnston
- (**website**) clarify wheel build output location (#277) - ([07d0061](https://github.com/ibm/ado/commit/07d0061d6596d4529f3ac3988be680af1f7a0329)) - Vassilis Vassiliadis
#### Build system
- (**deps**) update dependencies (#282) - ([4b4d8c2](https://github.com/ibm/ado/commit/4b4d8c2f0643be4a11ab5cbb64a0da03df26e63c)) - Alessandro Pomponio

## [1.2.4](https://github.com/ibm/ado/compare/1.2.3..1.2.4) - 2025-12-01
#### Features
- (**cli**) support shorthands for resources (#245) - ([39d4931](https://github.com/ibm/ado/commit/39d49315bbf1a5d41a836fb7dffc57ca6ce5922f)) - Alessandro Pomponio
- (**core**) enable actuators and operators to use Rich progress indicators (#248) - ([d930308](https://github.com/ibm/ado/commit/d9303082db39579b822e882dab3f4a8fada7b235)) - Michael Johnston
- (**group_samplers**) improve performance for group generator sampler type (#229) - ([7a79a23](https://github.com/ibm/ado/commit/7a79a23e2e62d9ef57ac70eaf146b4b015297f0c)) - Christian Pinto
#### Bug Fixes
- (**core**) show entities missing/unmeasured (#254) - ([f14ea2b](https://github.com/ibm/ado/commit/f14ea2b24a49b942590c9eeaac2c1de71c1d9d8e)) - Michael Johnston
- (**custom_experiments**) detect if custom experiment returns unexpected properties (#250) - ([89375e2](https://github.com/ibm/ado/commit/89375e21987d1de98e64f2b14b4504a12d248abb)) - Michael Johnston
- (**vllm_performance**) Avoid starvation of measurements requests (#249) - ([cd99105](https://github.com/ibm/ado/commit/cd991054ea66d6f6e92e424961dec00d7e88bb31)) - Christian Pinto
- (**vllm_performance**) example entity space was incompatible with measurement space (#240) - ([fe43998](https://github.com/ibm/ado/commit/fe43998d67f871a5eff15e7489006ce8706ab5d6)) - Michael Johnston
#### Documentation
- (**changelog**) update changelog (#267) - ([c7e1ee7](https://github.com/ibm/ado/commit/c7e1ee748eb6436d3d0171bdc034cda39c342ea6)) - Alessandro Pomponio
#### Build system
- (**deps**) update dependencies (#258) - ([abe438b](https://github.com/ibm/ado/commit/abe438b814708d2040685bcff0bfe2a5cd7937a0)) - Alessandro Pomponio
#### Refactoring
- (**cli**) remove HiddenSingularChoice (#246) - ([6edcc47](https://github.com/ibm/ado/commit/6edcc47d503adcecb7d1d5b11d09681ad51536f2)) - Alessandro Pomponio
- (**core**) unify operation execution pathways and remove unused logic (#220) - ([13f2259](https://github.com/ibm/ado/commit/13f2259cf70f1240b1c65682716e20a2c70d1710)) - Michael Johnston
- (**core**) do not change signature of functions decorated with custom_experiment (#261) - ([ece2f7d](https://github.com/ibm/ado/commit/ece2f7de62b68e20992e90cdc2244084b907a239)) - Michael Johnston

- - -

## [1.2.3](https://github.com/ibm/ado/compare/1.2.2..1.2.3) - 2025-11-21
#### Bug Fixes
- (**build**) regenerate lockfile (#239) - ([427bae4](https://github.com/ibm/ado/commit/427bae4ee1d3e397046578706088a7413f83fa3a)) - Alessandro Pomponio
- (**core**) do not discard operation outputs when shutdown is set (#219) - ([ac7c932](https://github.com/ibm/ado/commit/ac7c932ff486643656c3a2b651a77d2fafdcc576)) - Michael Johnston
- (**docs**) use correct indentation in sublists (#227) - ([296aed2](https://github.com/ibm/ado/commit/296aed2853e39b8c6dfd3945f0c304064830e5f5)) - Michael Johnston
- (**vllm_performance**) k8s resource not marked for cleaning on exception (#230) - ([48ece77](https://github.com/ibm/ado/commit/48ece775dc1d36fdaa81e05aa743d9ee102dbfe6)) - Michael Johnston
#### Documentation
- (**changelog**) update changelog (#228) - ([b637931](https://github.com/ibm/ado/commit/b6379319668592a4f4a5dea606c5ae1baa95dcb2)) - Alessandro Pomponio
- (**website**) update remote raycluster execution (#221) - ([3120648](https://github.com/ibm/ado/commit/312064890f010c03675ebe9e9dba8c4ab7a16ff5)) - Michael Johnston
#### Build system
- (**core**) update dependencies (#222) - ([f2bea1f](https://github.com/ibm/ado/commit/f2bea1f6fdc5de172b15a174b21be02377c64f19)) - Alessandro Pomponio
- (**deps**) move scipy dependency to ado-ray-tune (#231) - ([322a866](https://github.com/ibm/ado/commit/322a866b78550d3331ebfeb4ac9845c2893e3beb)) - Michael Johnston
- (**vllm_performance**) bump vLLM version to 0.11.1 (#223) - ([73768cc](https://github.com/ibm/ado/commit/73768ccd5793cc034a937ab8ea041c908fdb4b2e)) - Christian Pinto
- create test dependency group in pyproject (#215) - ([adb15eb](https://github.com/ibm/ado/commit/adb15eb81024bd51ba8ddeb08a5728f9643de6e8)) - Michael Johnston
#### Refactoring
- (**core**) split orchestrate into submodules (#217) - ([730ea4b](https://github.com/ibm/ado/commit/730ea4baeb32b28628645f5549c1e662f682cf6f)) - Michael Johnston
- (**linting**) use markdownlint-cli2 configuration file (#226) - ([e43209e](https://github.com/ibm/ado/commit/e43209ebd104cc1f4507ab77938b88c92d3c51fb)) - Alessandro Pomponio

- - -

## [1.2.2](https://github.com/ibm/ado/compare/1.2.1..1.2.2) - 2025-11-13
#### Bug Fixes
- (**ado-core**) Enable decorated experiments with ray.remote (#213) - ([ed6189d](https://github.com/ibm/ado/commit/ed6189d8afaca32e640491eb16ea94a26ef95e64)) - Michael Johnston
- (**vllm_performance**) improve error handling (#218) - ([3262f04](https://github.com/ibm/ado/commit/3262f040c71f1f534ce75a2c90c1329d8aaf47fc)) - Christian Pinto
- (**vllm_performance**) Fixing various bugs with the vllm_perf actuator (#210) - ([b2191c6](https://github.com/ibm/ado/commit/b2191c61dd48d8faa7928f36cdf2bf56d20fdc5e)) - Christian Pinto
#### Documentation
- (**vllm_performance**) update website docs (#211) - ([1522cb9](https://github.com/ibm/ado/commit/1522cb96a25d261cc4ecb76d0533d2496d51b03a)) - Michael Johnston
- clarify commit and PR title guidelines (#207) - ([d3e41ea](https://github.com/ibm/ado/commit/d3e41eaf72eae9b20c0a68dd75aa6afa61899c95)) - Alessandro Pomponio
#### Refactoring
- (**vllm_performance**) change experiment name (#216) - ([83d27de](https://github.com/ibm/ado/commit/83d27ded869461de564f162401053bedb2eedfa9)) - Michael Johnston

- - -

## [1.2.1](https://github.com/ibm/ado/compare/1.2.0..1.2.1) - 2025-11-06
#### Miscellaneous Chores
- (**deps**) update vllm_performance's lockfile (#201) - ([44109d9](https://github.com/ibm/ado/commit/44109d90cfcae05ab3d32629dace9ca0de2b08e8)) - Alessandro Pomponio

- - -

## [1.2.0](https://github.com/ibm/ado/compare/1.1.0..1.2.0) - 2025-11-06
#### Features
- add support for more granite-4.0 models (#199) - ([34f08b7](https://github.com/ibm/ado/commit/34f08b7a691fa03e95185a402e647cde328845cc)) - Vassilis Vassiliadis
- support granite-4.0 models (#192) - ([8ff2070](https://github.com/ibm/ado/commit/8ff2070682bfed1b123514e5a546e92dd766f849)) - Vassilis Vassiliadis
- decorator for custom_experiments (#154) - ([09b09b8](https://github.com/ibm/ado/commit/09b09b89b662387ec0250534d6db34e5d176a613)) - Michael Johnston
- add support for --use-latest flag in ado describe space (#176) - ([9cd90a0](https://github.com/ibm/ado/commit/9cd90a043cafefef5c7cead756c3f9ef5cbee9db)) - Alessandro Pomponio
- support --use-latest flag in ado show commands (#166) - ([2ba5dc1](https://github.com/ibm/ado/commit/2ba5dc185805907298f7cb585d41d7a28b2f856d)) - Alessandro Pomponio
- add default sample store and --use-default-sample-store (#157) - ([25dd190](https://github.com/ibm/ado/commit/25dd19042dd6e15587a95b87bd8c62653801c1a4)) - Alessandro Pomponio
- add --with-latest flag in ado create to support reusing of latest identifiers (#152) - ([e8fb48b](https://github.com/ibm/ado/commit/e8fb48b7951bc7c9f1912f5c37ab2dadbc63b875)) - Alessandro Pomponio
- enable copywrite pre-commit hook (#155) - ([07ef0ac](https://github.com/ibm/ado/commit/07ef0ac8a614953640c926ba56ddd08b7814ec81)) - Alessandro Pomponio
- record identifier of the latest resource created using ado create (#149) - ([f5f1583](https://github.com/ibm/ado/commit/f5f1583f318cbe203bdaebb4246a7b42e7b85209)) - Alessandro Pomponio
- implement a RuntimeEnvPlugin to guide installation order of packages (#126) - ([96ecf03](https://github.com/ibm/ado/commit/96ecf038e14307cbfc4a2fcd23347af0535e6729)) - Vassilis Vassiliadis
- open categorical variable type (#118) - ([acbfa7c](https://github.com/ibm/ado/commit/acbfa7c853ad7968ed558ac822108fa95c4df8cc)) - Michael Johnston
- support more models (#124) - ([77f314c](https://github.com/ibm/ado/commit/77f314ce491803a5f8ed15a358a8a627ddaf12bc)) - Vassilis Vassiliadis
- support live display of measurement results during operations (#122) - ([7fbd365](https://github.com/ibm/ado/commit/7fbd36530c867fb8bd0eaecd891fd5ace6115367)) - Alessandro Pomponio
- add constitutive properties to show entities operation (#116) - ([9ce2735](https://github.com/ibm/ado/commit/9ce273582789bd3c5fd7a49fbcd057230f139d10)) - Alessandro Pomponio
- add run_experiment script (#77) - ([e4acec3](https://github.com/ibm/ado/commit/e4acec31a8924a1a9a67a4acf1110e8c1792f257)) - Michael Johnston
#### Bug Fixes
- (**regression**) update pre-commit hooks (#198) - ([902e621](https://github.com/ibm/ado/commit/902e6210c3c44609adf2046757bbba7873b53785)) - Alessandro Pomponio
- (**run_experiment**) pass actuator configuration ids (#117) - ([1e0820f](https://github.com/ibm/ado/commit/1e0820ffbb7f6962462ca8292e8ed06d12eaf6bd)) - Michael Johnston
- (**vllm_performance**) cap deployment name length and update cli flag (#195) - ([ff245ca](https://github.com/ibm/ado/commit/ff245caf00e834a40a4f407d9135cabd363ca2db)) - Christian Pinto
- log entity validation errors only when verbose output is enabled (#197) - ([5866ded](https://github.com/ibm/ado/commit/5866ded9a6ef501d8b24de323ae2dcb2e9714f32)) - Michael Johnston
- the OrderedPip RayRuntimeEnv plugin and the SFTTrainer code that uses it (#186) - ([dee913e](https://github.com/ibm/ado/commit/dee913e2f432448b9d18a22d826828d773c6dcef)) - Vassilis Vassiliadis
- Fixed bug in validate_entity for run_experiment script, improved logging (#182) - ([6ae8999](https://github.com/ibm/ado/commit/6ae89998baececc30f9b741c9d8a4eed70fe1278)) - Christian Pinto
- parameterization of custom experiments (#179) - ([f427683](https://github.com/ibm/ado/commit/f4276832f8fe424ea477be1ccd816abc8102911f)) - Michael Johnston
- remove active field in mysql onboarding script (#177) - ([a4e1bd8](https://github.com/ibm/ado/commit/a4e1bd8832b84e5cbc561c947c3d87b09185eae1)) - Alessandro Pomponio
- make datetime timezone aware in ado show summary (#139) - ([e08e10d](https://github.com/ibm/ado/commit/e08e10d116e726edee10df8669974ff1ea578524)) - Michael Johnston
- accessing non-existent field (#137) - ([4387b33](https://github.com/ibm/ado/commit/4387b33827223c7f4831092c612328b2aea19298)) - Michael Johnston
- fetch entity from db if not in the sample store cache (#121) - ([f418075](https://github.com/ibm/ado/commit/f4180750a5173b7b591f1cb12ab950b5a7774c9e)) - Alessandro Pomponio
- support isSubDomain with BINARY_VARIABLE_TYPE (#106) - ([34d2077](https://github.com/ibm/ado/commit/34d207726bb0acf9389ca44be7a67d2e3d1f0603)) - Michael Johnston
#### Documentation
- (**website**) update examples (#168) - ([ea15c4b](https://github.com/ibm/ado/commit/ea15c4bb13a1a559dd340d3e51424fce931e9ade)) - Michael Johnston
- (**website**) update documentation to latest state (#175) - ([0177a2e](https://github.com/ibm/ado/commit/0177a2e91cca3a395c2d0d1dd4152abcc16d8407)) - Alessandro Pomponio
- (**website**) vllm-performance actuator (#113) - ([9267d3b](https://github.com/ibm/ado/commit/9267d3b0234597a63e47dc49fdc14f233798e8e2)) - Michael Johnston
- explain how to configure ServiceAccount permissions for RayClusters (#196) - ([c53bcdb](https://github.com/ibm/ado/commit/c53bcdbf44ad0d3b6996241de74833149ba2a629)) - Christian Pinto
- change fms_hf_tuning_version to 2.8.2 for the finetune-locally example (#138) - ([33e6424](https://github.com/ibm/ado/commit/33e6424f631a5ea2eda4517b36ef33a244c2fc61)) - Vassilis Vassiliadis
- fix typo in vllm-performance-full.md (#136) - ([6a09275](https://github.com/ibm/ado/commit/6a0927505001ce69eeb7829b3c9d37f9ee8c7fef)) - Vassilis Vassiliadis
- fix vllm-performance install docs (#134) - ([fda6501](https://github.com/ibm/ado/commit/fda6501328c8f903c98f09d660d64ff6c29f9173)) - Michael Johnston
#### Tests
- use uv runner using lockfile (#129) - ([2ea54b8](https://github.com/ibm/ado/commit/2ea54b8f89f263458865774f41a81b2446f39f6b)) - Alessandro Pomponio
#### Build system
- ensure container images use locked dependencies (#142) - ([3411ce3](https://github.com/ibm/ado/commit/3411ce3a7df8a4390d012b1d589bdded549416e1)) - Alessandro Pomponio
#### Refactoring
- rename --with-latest flag to --use-latest (#164) - ([54e7721](https://github.com/ibm/ado/commit/54e7721c4d364f46ec70fb9699f0c0746ff094e4)) - Alessandro Pomponio
#### Miscellaneous Chores
- (**deps**) update dependencies (#193) - ([7195177](https://github.com/ibm/ado/commit/7195177fc1a6514205a8a7e3666ef2ef816a5fdf)) - Alessandro Pomponio
- (**deps**) update dependencies (#189) - ([c70ead6](https://github.com/ibm/ado/commit/c70ead64ac9f285db6136ebe0526d907456dba41)) - Alessandro Pomponio
- (**deps**) update ray to v2.51.0 (#173) - ([43cfc66](https://github.com/ibm/ado/commit/43cfc66e5f0a35213fa402984961faf80eacbf2b)) - Alessandro Pomponio
- (**deps**) update dependencies (#171) - ([82fe780](https://github.com/ibm/ado/commit/82fe780937d620e32f454cec81006ca333a91828)) - Alessandro Pomponio
- (**deps**) update dependencies (#158) - ([f5194d1](https://github.com/ibm/ado/commit/f5194d117066655c123a47d9b44bcc3c452b2834)) - Alessandro Pomponio
- (**deps**) upgrade dependencies (#140) - ([01cb262](https://github.com/ibm/ado/commit/01cb2622efc74c689c3a3350c326baaac475072d)) - Alessandro Pomponio
- (**deps**) update dependencies (#107) - ([85add1c](https://github.com/ibm/ado/commit/85add1cdfc7ffb32e5a5037520275e9c6af219d6)) - Alessandro Pomponio
- (**deps**) update dependencies (#96) - ([adc7b9f](https://github.com/ibm/ado/commit/adc7b9feb7de3afea9d18fd83c68180bbc149126)) - Alessandro Pomponio
- (**vllm-performance**) update dependencies (#108) - ([8b1d91e](https://github.com/ibm/ado/commit/8b1d91ea5002d9d3d46a19bd780469eff55dd5df)) - Alessandro Pomponio
- update pre-commit hooks (#194) - ([fc15d72](https://github.com/ibm/ado/commit/fc15d72c3983ee1b629f3aa4a651455af62ba81a)) - Alessandro Pomponio
- remove upgrade validator for randomwalk parameters (#188) - ([cee5a42](https://github.com/ibm/ado/commit/cee5a42056c31202afa120db793dd2586884a2d0)) - Alessandro Pomponio
- make target the default property format for ado show entities (#161) - ([ea4d081](https://github.com/ibm/ado/commit/ea4d0818f1786fdc7fdb12716850f909bc5d96a0)) - Alessandro Pomponio

- - -

## [1.1.0](https://github.com/ibm/ado/compare/1.0.1..1.1.0) - 2025-10-03
#### Features
- add info message if actuator does not have experiments (#80) - ([fe40792](https://github.com/ibm/ado/commit/fe407923de14bf867560ab4aaf67f0be3bd70c53)) - Alessandro Pomponio
- add support for booleans and null values in sqlite field querying (#82) - ([663fa0c](https://github.com/ibm/ado/commit/663fa0c9c89adeed38586ee8ee7ca8d955dc8479)) - Alessandro Pomponio
- dump default values by default when getting contexts (#74) - ([6464f3a](https://github.com/ibm/ado/commit/6464f3a5cb15476eab2cf1be5fc9e59c189d1048)) - Alessandro Pomponio
- implement REST API MVP (#47) - ([9c6b078](https://github.com/ibm/ado/commit/9c6b0787ccaae4fc164921610b7fb42442a2c880)) - Alessandro Pomponio
- add support for fms-hf-tuning==3.0.0 in SFTTrainer experiments (#42) - ([a4fd319](https://github.com/ibm/ado/commit/a4fd319178e0ab800f1efca27f7ff6d8004db271)) - Vassilis Vassiliadis
- support auto_stop_method for SFTTrainer experiments (#27) - ([6be963f](https://github.com/ibm/ado/commit/6be963fa91c75e4df7ac8ef9db99657ec2934f6e)) - Vassilis Vassiliadis
- allow specifying custom sampler class for use with `random_walk` operator (#26) - ([1c62218](https://github.com/ibm/ado/commit/1c622189a10ccf975492d40e29c757678db3055a)) - Michael Johnston
- setting aim_db to None configures SFTTrainer to use an ephemeral AIM repository (#24) - ([7f731c8](https://github.com/ibm/ado/commit/7f731c821455414d4e20d5a08659b55ebc7b3634)) - Vassilis Vassiliadis
- support llava-v1.6-mistral-7b (#15) - ([fb78848](https://github.com/ibm/ado/commit/fb7884831d72d663c2c8386dd3ae3c27b7b76c5b)) - Vassilis Vassiliadis
#### Bug Fixes
- (**build**) introduce build-system section (#14) - ([dd12659](https://github.com/ibm/ado/commit/dd12659d057ef4dd61ac12c8f5cd0e152cb67084)) - Alessandro Pomponio
- (**docs**) fix typos  (#72) - ([d9c09fb](https://github.com/ibm/ado/commit/d9c09fb60ced3f3ad4e9b8741e5d15058eac27d8)) - Daniele Lotito
- (**docs**) update path for local context (#10) - ([ea8662f](https://github.com/ibm/ado/commit/ea8662f5b747e7a850a51b29b075b0ca56c35f3d)) - Alessandro Pomponio
- (**style**) apply fixes for RUF059 unused-unpacked-variable (#61) - ([e0993eb](https://github.com/ibm/ado/commit/e0993ebcda2a62254004c0eaaf545b7887a82d28)) - Alessandro Pomponio
- minor issues (#89) - ([30e1173](https://github.com/ibm/ado/commit/30e11733b09b6984c1d955a4769d821863c67f27)) - Michael Johnston
- retrieving the result of an experiment request from the ado REST API (#88) - ([3625fab](https://github.com/ibm/ado/commit/3625fab83707445151ff0349d5f10db5e663c486)) - Vassilis Vassiliadis
- ensure ado get -o json works (#84) - ([cae4081](https://github.com/ibm/ado/commit/cae4081b1fc20eac684aa54da502dce0e01e1def)) - Alessandro Pomponio
- ensure simulated JSON_CONTAINS works on SQLite (#78) - ([d4afafb](https://github.com/ibm/ado/commit/d4afafbfa1de7cf7bf5011e2d05093673462eca5)) - Alessandro Pomponio
- ensure sample store identifiers cannot be parsed as floats (#76) - ([0e53b56](https://github.com/ibm/ado/commit/0e53b56aad190a25f226a6a53a27f796cd747f7d)) - Alessandro Pomponio
- use correct variable in ado template operation (#73) - ([ccaf27f](https://github.com/ibm/ado/commit/ccaf27f1a04d5c8e8626048f78ef0d6e08a957c3)) - Michael Johnston
- calculating the throughput for SFTTrainer experiments (#70) - ([73c5a94](https://github.com/ibm/ado/commit/73c5a9458528c60ae2146e62884c20a9c8e17f4c)) - Vassilis Vassiliadis
- measurement request serialization (#56) - ([f658b8d](https://github.com/ibm/ado/commit/f658b8d66a81931ed9dc15106ee3badad15ba890)) - Michael Johnston
- measuring properties in the example_actuator (#45) - ([8efd40a](https://github.com/ibm/ado/commit/8efd40a7776af9b1b52f06e4b9c16c6030435f3e)) - Vassilis Vassiliadis
- configuring Trainer to exit a training job early (#41) - ([aca6167](https://github.com/ibm/ado/commit/aca61672e47fe3ead1d23176769375748e61449d)) - Vassilis Vassiliadis
- Potential access of unset var on Exception (#36) - ([a05193e](https://github.com/ibm/ado/commit/a05193e5bdac4fc58d0d9ece0174740f9ec84c75)) - Michael Johnston
#### Documentation
- update metastore query docs (#69) - ([d690e83](https://github.com/ibm/ado/commit/d690e835a7fb6c35fa39f6d1d1d14bbdf6d9de88)) - Michael Johnston
- improve docs for the random walk operator (#53) - ([35abb8e](https://github.com/ibm/ado/commit/35abb8ea48ac6ff1672c4cf8f44efa4f966d6680)) - Daniele Lotito
- add acknowledgements (#50) - ([f449223](https://github.com/ibm/ado/commit/f44922366845f90a4b4db3ef6470935fbe465de9)) - Alessandro Pomponio
- make sure developing and contributing instructions are complete (#40) - ([cbbac07](https://github.com/ibm/ado/commit/cbbac0759291e73117d87cb9b012dadda05da3a2)) - Alessandro Pomponio
#### Tests
- ensure example_actuator is tested in CI (#48) - ([dfde15f](https://github.com/ibm/ado/commit/dfde15f30eab6594cbcd7a60a8a3e93a2fb43cf8)) - Alessandro Pomponio
#### Build system
- use hatchling in example custom experiments (#67) - ([f64adf3](https://github.com/ibm/ado/commit/f64adf36fa930a95114470b1191db58e9208cab8)) - Michael Johnston
- remove torch from the list of SFTTrainer dependencies (#38) - ([cea1150](https://github.com/ibm/ado/commit/cea1150e328aa4af4fc252206594352263579ecc)) - Vassilis Vassiliadis
- link readme in ado-sfttrainer (#12) - ([876a20a](https://github.com/ibm/ado/commit/876a20a9dec2e296466393f26724bfa6bee92311)) - Alessandro Pomponio
#### Refactoring
- Property and PropertyValue models (#49) - ([ac0c841](https://github.com/ibm/ado/commit/ac0c841ad5c84743910cd0ae9e5b7d337e838f31)) - Michael Johnston
- remove script dependency from vLLM Performance actuator (#25) - ([e1ed60f](https://github.com/ibm/ado/commit/e1ed60f363a57ad1ea11319a1002d8fbdd145eee)) - Srikumar Venugopal
- drop support for ax in the Ray Tune operator (#33) - ([6e7a903](https://github.com/ibm/ado/commit/6e7a903b720297802c9da89ad9fef5d81dab4e53)) - Alessandro Pomponio
#### Miscellaneous Chores
- (**deps**) update dependencies (#90) - ([e7ebc73](https://github.com/ibm/ado/commit/e7ebc733a13071fbaa4f286f997b978f6c9c976f)) - Alessandro Pomponio
- (**deps**) update dependencies (#79) - ([63cbf7f](https://github.com/ibm/ado/commit/63cbf7f25e23f578f077ba567bbe64a898bacb96)) - Alessandro Pomponio
- (**deps**) update dependencies (#66) - ([cf4e8b3](https://github.com/ibm/ado/commit/cf4e8b3c1d412dc41b34b4dfbd994bff33c47792)) - Alessandro Pomponio
- (**deps**) update dependencies (#59) - ([3f895c0](https://github.com/ibm/ado/commit/3f895c03efae420e866c7a55063ff27c06122e84)) - Alessandro Pomponio
- (**deps**) do not pin numpy<2 anymore (#28) - ([27c7c81](https://github.com/ibm/ado/commit/27c7c811c6550c9c4a249e120965ce5b040c0aee)) - Alessandro Pomponio
- (**deps**) update dependencies (#18) - ([f16fcde](https://github.com/ibm/ado/commit/f16fcdedad4df1018eddc1ee17b70e4a47e24a03)) - Alessandro Pomponio
- Configure Renovate (#1) - ([1346ac5](https://github.com/ibm/ado/commit/1346ac5b05f407e57c2456a8cf3debb8f0190f57)) - renovate[bot]
- update security reporting (#75) - ([ae0aee7](https://github.com/ibm/ado/commit/ae0aee7047ce28001bf61753de06abf793ee669a)) - Alessandro Pomponio
- update mend configuration (#62) - ([334027d](https://github.com/ibm/ado/commit/334027de4fdafd5200ec7f61ebbc7066dfe08016)) - Alessandro Pomponio
- add funding acknowledgements (#51) - ([2ccfb96](https://github.com/ibm/ado/commit/2ccfb96cb77c768f4b914fe17fa018e30f15807c)) - Alessandro Pomponio
- website fixes (#19) - ([884d95c](https://github.com/ibm/ado/commit/884d95c3b829952b90ab58f433d4e7afbfed5f2a)) - Michael Johnston
#### Style
- lint markdown files (#23) - ([8a6aa42](https://github.com/ibm/ado/commit/8a6aa420a4188db2e79d7ca43b6dac1be3524314)) - Alessandro Pomponio
- enable ruff's SIM linter (#21) - ([6728c7b](https://github.com/ibm/ado/commit/6728c7b89519222012a8a086a1e360be0c2a0da9)) - Alessandro Pomponio
- apply ruff's UP linter (#17) - ([e16a83a](https://github.com/ibm/ado/commit/e16a83aded0326a841652297219304c36ac9990d)) - Alessandro Pomponio

- - -

## [1.0.1](https://github.com/ibm/ado/compare/1.0.0..1.0.1) - 2025-09-01
#### Build system
- rename ado-base to ado-core (#6) - ([1f16068](https://github.com/ibm/ado/commit/1f160680f646153f4a98030dfc25f858e579e31e)) - Alessandro Pomponio
#### Miscellaneous Chores
- remove upgrade validators (#8) - ([a537516](https://github.com/ibm/ado/commit/a537516873149124c2b866c6c718d49938dd8502)) - Alessandro Pomponio

- - -

## [1.0.0](https://github.com/ibm/ado/compare/294a321fadf06a190209f1eda70868e66d2d4884..1.0.0) - 2025-08-29
#### Features
- initial commit - ([7401b9d](https://github.com/ibm/ado/commit/7401b9d6a169373fb6542ef8aa9ab363605a151e)) - Alessandro Pomponio
#### Documentation
- fix broken links (#4) - ([a0aa321](https://github.com/ibm/ado/commit/a0aa32119aa88dcde92fa335885e8030ee783265)) - Vassilis Vassiliadis
- replace references to the old repository with the refs to the new ones (#2) - ([9fae2bf](https://github.com/ibm/ado/commit/9fae2bfa1010589f9b28cd5270c16f03763184c4)) - Vassilis Vassiliadis
#### Build system
- add dynamic versioning to actuators and operators (#3) - ([3ab9c4d](https://github.com/ibm/ado/commit/3ab9c4d51fd64030c167cfc5a9b7e19402dd8107)) - Vassilis Vassiliadis



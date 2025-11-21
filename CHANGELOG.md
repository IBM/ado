## 1.2.2 - 2025-11-20
#### Bug Fixes
- (**ado-core**) Enable decorated experiments with ray.remote (#213) - (ed6189d) - Michael Johnston
- (**vllm_performance**) improve error handling (#218) - (3262f04) - Christian Pinto
- (**vllm_performance**) Fixing various bugs with the vllm_perf actuator (#210) - (b2191c6) - Christian Pinto
#### Documentation
- (**vllm_performance**) update website docs (#211) - (1522cb9) - Michael Johnston
- clarify commit and PR title guidelines (#207) - (d3e41ea) - Alessandro Pomponio
#### Refactoring
- (**vllm_performance**) change experiment name (#216) - (83d27de) - Michael Johnston

- - -

## 1.2.1 - 2025-11-20
#### Miscellaneous Chores
- (**deps**) update vllm_performance's lockfile (#201) - (44109d9) - Alessandro Pomponio

- - -

## 1.2.0 - 2025-11-20
#### Features
- add support for more granite-4.0 models (#199) - (34f08b7) - Vassilis Vassiliadis
- support granite-4.0 models (#192) - (8ff2070) - Vassilis Vassiliadis
- decorator for custom_experiments (#154) - (09b09b8) - Michael Johnston
- add support for --use-latest flag in ado describe space (#176) - (9cd90a0) - Alessandro Pomponio
- support --use-latest flag in ado show commands (#166) - (2ba5dc1) - Alessandro Pomponio
- add default sample store and --use-default-sample-store (#157) - (25dd190) - Alessandro Pomponio
- add --with-latest flag in ado create to support reusing of latest identifiers (#152) - (e8fb48b) - Alessandro Pomponio
- enable copywrite pre-commit hook (#155) - (07ef0ac) - Alessandro Pomponio
- record identifier of the latest resource created using ado create (#149) - (f5f1583) - Alessandro Pomponio
- implement a RuntimeEnvPlugin to guide installation order of packages (#126) - (96ecf03) - Vassilis Vassiliadis
- open categorical variable type (#118) - (acbfa7c) - Michael Johnston
- support more models (#124) - (77f314c) - Vassilis Vassiliadis
- support live display of measurement results during operations (#122) - (7fbd365) - Alessandro Pomponio
- add constitutive properties to show entities operation (#116) - (9ce2735) - Alessandro Pomponio
- add run_experiment script (#77) - (e4acec3) - Michael Johnston
#### Bug Fixes
- (**regression**) update pre-commit hooks (#198) - (902e621) - Alessandro Pomponio
- (**run_experiment**) pass actuator configuration ids (#117) - (1e0820f) - Michael Johnston
- (**vllm_performance**) cap deployment name length and update cli flag (#195) - (ff245ca) - Christian Pinto
- log entity validation errors only when verbose output is enabled (#197) - (5866ded) - Michael Johnston
- the OrderedPip RayRuntimeEnv plugin and the SFTTrainer code that uses it (#186) - (dee913e) - Vassilis Vassiliadis
- Fixed bug in validate_entity for run_experiment script, improved logging (#182) - (6ae8999) - Christian Pinto
- parameterization of custom experiments (#179) - (f427683) - Michael Johnston
- remove active field in mysql onboarding script (#177) - (a4e1bd8) - Alessandro Pomponio
- make datetime timezone aware in ado show summary (#139) - (e08e10d) - Michael Johnston
- accessing non-existent field (#137) - (4387b33) - Michael Johnston
- fetch entity from db if not in the sample store cache (#121) - (f418075) - Alessandro Pomponio
- support isSubDomain with BINARY_VARIABLE_TYPE (#106) - (34d2077) - Michael Johnston
#### Documentation
- (**website**) update examples (#168) - (ea15c4b) - Michael Johnston
- (**website**) update documentation to latest state (#175) - (0177a2e) - Alessandro Pomponio
- (**website**) vllm-performance actuator (#113) - (9267d3b) - Michael Johnston
- explain how to configure ServiceAccount permissions for RayClusters (#196) - (c53bcdb) - Christian Pinto
- change fms_hf_tuning_version to 2.8.2 for the finetune-locally example (#138) - (33e6424) - Vassilis Vassiliadis
- fix typo in vllm-performance-full.md (#136) - (6a09275) - Vassilis Vassiliadis
- fix vllm-performance install docs (#134) - (fda6501) - Michael Johnston
#### Tests
- use uv runner using lockfile (#129) - (2ea54b8) - Alessandro Pomponio
#### Build system
- ensure container images use locked dependencies (#142) - (3411ce3) - Alessandro Pomponio
#### Refactoring
- rename --with-latest flag to --use-latest (#164) - (54e7721) - Alessandro Pomponio
#### Miscellaneous Chores
- (**deps**) update dependencies (#193) - (7195177) - Alessandro Pomponio
- (**deps**) update dependencies (#189) - (c70ead6) - Alessandro Pomponio
- (**deps**) update ray to v2.51.0 (#173) - (43cfc66) - Alessandro Pomponio
- (**deps**) update dependencies (#171) - (82fe780) - Alessandro Pomponio
- (**deps**) update dependencies (#158) - (f5194d1) - Alessandro Pomponio
- (**deps**) upgrade dependencies (#140) - (01cb262) - Alessandro Pomponio
- (**deps**) update dependencies (#107) - (85add1c) - Alessandro Pomponio
- (**deps**) update dependencies (#96) - (adc7b9f) - Alessandro Pomponio
- (**vllm-performance**) update dependencies (#108) - (8b1d91e) - Alessandro Pomponio
- update pre-commit hooks (#194) - (fc15d72) - Alessandro Pomponio
- remove upgrade validator for randomwalk parameters (#188) - (cee5a42) - Alessandro Pomponio
- make target the default property format for ado show entities (#161) - (ea4d081) - Alessandro Pomponio

- - -

## 1.1.0 - 2025-11-20
#### Features
- add info message if actuator does not have experiments (#80) - (fe40792) - Alessandro Pomponio
- add support for booleans and null values in sqlite field querying (#82) - (663fa0c) - Alessandro Pomponio
- dump default values by default when getting contexts (#74) - (6464f3a) - Alessandro Pomponio
- implement REST API MVP (#47) - (9c6b078) - Alessandro Pomponio
- add support for fms-hf-tuning==3.0.0 in SFTTrainer experiments (#42) - (a4fd319) - Vassilis Vassiliadis
- support auto_stop_method for SFTTrainer experiments (#27) - (6be963f) - Vassilis Vassiliadis
- allow specifying custom sampler class for use with `random_walk` operator (#26) - (1c62218) - Michael Johnston
- setting aim_db to None configures SFTTrainer to use an ephemeral AIM repository (#24) - (7f731c8) - Vassilis Vassiliadis
- support llava-v1.6-mistral-7b (#15) - (fb78848) - Vassilis Vassiliadis
#### Bug Fixes
- (**build**) introduce build-system section (#14) - (dd12659) - Alessandro Pomponio
- (**docs**) fix typos  (#72) - (d9c09fb) - Daniele Lotito
- (**docs**) update path for local context (#10) - (ea8662f) - Alessandro Pomponio
- (**style**) apply fixes for RUF059 unused-unpacked-variable (#61) - (e0993eb) - Alessandro Pomponio
- minor issues (#89) - (30e1173) - Michael Johnston
- retrieving the result of an experiment request from the ado REST API (#88) - (3625fab) - Vassilis Vassiliadis
- ensure ado get -o json works (#84) - (cae4081) - Alessandro Pomponio
- ensure simulated JSON_CONTAINS works on SQLite (#78) - (d4afafb) - Alessandro Pomponio
- ensure sample store identifiers cannot be parsed as floats (#76) - (0e53b56) - Alessandro Pomponio
- use correct variable in ado template operation (#73) - (ccaf27f) - Michael Johnston
- calculating the throughput for SFTTrainer experiments (#70) - (73c5a94) - Vassilis Vassiliadis
- measurement request serialization (#56) - (f658b8d) - Michael Johnston
- measuring properties in the example_actuator (#45) - (8efd40a) - Vassilis Vassiliadis
- configuring Trainer to exit a training job early (#41) - (aca6167) - Vassilis Vassiliadis
- Potential access of unset var on Exception (#36) - (a05193e) - Michael Johnston
#### Documentation
- update metastore query docs (#69) - (d690e83) - Michael Johnston
- improve docs for the random walk operator (#53) - (35abb8e) - Daniele Lotito
- add acknowledgements (#50) - (f449223) - Alessandro Pomponio
- make sure developing and contributing instructions are complete (#40) - (cbbac07) - Alessandro Pomponio
#### Tests
- ensure example_actuator is tested in CI (#48) - (dfde15f) - Alessandro Pomponio
#### Build system
- use hatchling in example custom experiments (#67) - (f64adf3) - Michael Johnston
- remove torch from the list of SFTTrainer dependencies (#38) - (cea1150) - Vassilis Vassiliadis
- link readme in ado-sfttrainer (#12) - (876a20a) - Alessandro Pomponio
#### Refactoring
- Property and PropertyValue models (#49) - (ac0c841) - Michael Johnston
- remove script dependency from vLLM Performance actuator (#25) - (e1ed60f) - Srikumar Venugopal
- drop support for ax in the Ray Tune operator (#33) - (6e7a903) - Alessandro Pomponio
#### Miscellaneous Chores
- (**deps**) update dependencies (#90) - (e7ebc73) - Alessandro Pomponio
- (**deps**) update dependencies (#79) - (63cbf7f) - Alessandro Pomponio
- (**deps**) update dependencies (#66) - (cf4e8b3) - Alessandro Pomponio
- (**deps**) update dependencies (#59) - (3f895c0) - Alessandro Pomponio
- (**deps**) do not pin numpy<2 anymore (#28) - (27c7c81) - Alessandro Pomponio
- (**deps**) update dependencies (#18) - (f16fcde) - Alessandro Pomponio
- Configure Renovate (#1) - (1346ac5) - renovate[bot]
- update security reporting (#75) - (ae0aee7) - Alessandro Pomponio
- update mend configuration (#62) - (334027d) - Alessandro Pomponio
- add funding acknowledgements (#51) - (2ccfb96) - Alessandro Pomponio
- website fixes (#19) - (884d95c) - Michael Johnston
#### Style
- lint markdown files (#23) - (8a6aa42) - Alessandro Pomponio
- enable ruff's SIM linter (#21) - (6728c7b) - Alessandro Pomponio
- apply ruff's UP linter (#17) - (e16a83a) - Alessandro Pomponio

- - -

## 1.0.1 - 2025-11-20
#### Build system
- rename ado-base to ado-core (#6) - (1f16068) - Alessandro Pomponio
#### Miscellaneous Chores
- remove upgrade validators (#8) - (a537516) - Alessandro Pomponio

- - -

## 1.0.0 - 2025-11-20
#### Features
- initial commit - (7401b9d) - Alessandro Pomponio
#### Documentation
- fix broken links (#4) - (a0aa321) - Vassilis Vassiliadis
- replace references to the old repository with the refs to the new ones (#2) - (9fae2bf) - Vassilis Vassiliadis
#### Build system
- add dynamic versioning to actuators and operators (#3) - (3ab9c4d) - Vassilis Vassiliadis



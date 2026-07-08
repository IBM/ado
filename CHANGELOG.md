## [2.0.0](https://github.com/ibm/ado/compare/a25f83090fe28e144bda9c3065769a116ee0e639..2.0.0) - 2026-07-08
#### Features
- (**cli**) add ado show stats (#1105) - ([0cbd9bc](https://github.com/ibm/ado/commit/0cbd9bc433cc54105452d3081ba086ab0b598214)) - Alessandro Pomponio
- (**cli**) support -o stats in ado get datacontainer (#1091) - ([c72d428](https://github.com/ibm/ado/commit/c72d4282a30fad952fb26d18ec19d1c20f3fa0fb)) - Alessandro Pomponio
- (**cli**) support samplestores in ado get -o stats (#1085) - ([6cbc81f](https://github.com/ibm/ado/commit/6cbc81f4878d0788f0733abbf10ff7ba9423a6f9)) - Alessandro Pomponio
- (**cli**) add ado get space -o stats (#1082) - ([01a2d47](https://github.com/ibm/ado/commit/01a2d4788b821aaf0a8f0b6ced4c2f3905dc010a)) - Alessandro Pomponio
- (**cli**) add -o stats option to ado get operation (#1073) - ([6cbf5ad](https://github.com/ibm/ado/commit/6cbf5adaf7501402fb9aa536928a55de95ccd030)) - Alessandro Pomponio
- (**cli**) add ado show trace space/store (#1063) - ([b42aa2a](https://github.com/ibm/ado/commit/b42aa2a4b3bef3fb0dcbe714a27e2f2355165dc5)) - Alessandro Pomponio
- (**cli**) add ado show trace command (#1035) - ([57b5919](https://github.com/ibm/ado/commit/57b59190f7fa0960abd0c9018cd5fdd5dac0468a)) - Alessandro Pomponio
- (**cli**) support --patch and --patch-file flags in ado edit for non-interactive metadata patches (#906) - ([664b170](https://github.com/ibm/ado/commit/664b1703597d5b11bd79357b68804ca286df005c)) - Michael Johnston
- (**cli**) support --use-latest in ado get (#935) - ([7379fc8](https://github.com/ibm/ado/commit/7379fc8d7cdde0a4ea7b696039b261cbc51cdb3a)) - Alessandro Pomponio
- (**core**) add upgrade mechanism for orchestrator references in modules (#1187) - ([47a7977](https://github.com/ibm/ado/commit/47a79776cb5f826a742ae82220fa8cd945a53a13)) - Alessandro Pomponio
- (**core**) add StandardActuator base class for simple creation of actuator plugins (#955) - ([ff5a3d2](https://github.com/ibm/ado/commit/ff5a3d2024093a8b815e0ef3d618acb05d8201cd)) - Michael Johnston
- (**core**) improve versioning support for operators (#1124) - ([135e957](https://github.com/ibm/ado/commit/135e9579d4493abe3ea4625653532dc53206bc37)) - Michael Johnston
- (**core**) add experiment versioning support (#1043) - ([81c4055](https://github.com/ibm/ado/commit/81c40555df52fd1ee7131dc0b86a5028d44368e5)) - Michael Johnston
- (**core**) add ado provenance (#1095) - ([a43413e](https://github.com/ibm/ado/commit/a43413e59a440465da59e7516cdf931323845332)) - Michael Johnston
- (**core**) extend DiscoverySpaceStatistics with all stats from summary (#1104) - ([9f12b6b](https://github.com/ibm/ado/commit/9f12b6b5289bcffcb815ce753b144c386960ba16)) - Alessandro Pomponio
- (**core**) calculate datacontainer stats (#1087) - ([b7eb274](https://github.com/ibm/ado/commit/b7eb274b18dbc425fae92da8f973ea5d05dec73f)) - Alessandro Pomponio
- (**core**) calculate statistics for sample stores (#1077) - ([f92310c](https://github.com/ibm/ado/commit/f92310c8c9fbcbb63cdbdb1b0657999b98340b09)) - Alessandro Pomponio
- (**core**) calculate statistics for spaces (#1075) - ([64a4d3b](https://github.com/ibm/ado/commit/64a4d3b1021ff4ba592ba8de5189c77b5afe352c)) - Alessandro Pomponio
- (**core**) support multiple operation ids in operation_measurement_statistics (#1068) - ([25269a3](https://github.com/ibm/ado/commit/25269a3a32bb2c576e60b7b982a2c0b5884a5be1)) - Alessandro Pomponio
- (**core**) support multiple operation ids in measurement_requests_for_operation (#1062) - ([c8fd7ba](https://github.com/ibm/ado/commit/c8fd7ba6db16f5897c9c46800cb61f05163e700a)) - Alessandro Pomponio
- (**core**) add get_resources_by_relationship (#1052) - ([5f91a65](https://github.com/ibm/ado/commit/5f91a654dae7f39b3b08498196fbf677330416ac)) - Alessandro Pomponio
- (**core**) add function to calculate operation statistics (#1048) - ([cd11148](https://github.com/ibm/ado/commit/cd1114805be8a9dcd46af2b8e0fa82586e46c4fa)) - Alessandro Pomponio
- (**core**) add filtering for measurements in sample store (#1032) - ([e75ceb8](https://github.com/ibm/ado/commit/e75ceb8a33c9918ef51982aca36d484972b33421)) - Alessandro Pomponio
- (**core**) enable conditional disabling of plugin model validation in resources (#1028) - ([a77d2fc](https://github.com/ibm/ado/commit/a77d2fc7882319f5bc7139db8273569eaa2b278b)) - Michael Johnston
- (**core**) detect Ray measurement tasks that never start, and add remote runtime setup options (#998) - ([f46da72](https://github.com/ibm/ado/commit/f46da7208002d175f22f9c5708bd30c08f8aeed8)) - Michael Johnston
- (**core**) record distribution package provenance on resource creation (#1002) - ([b66b736](https://github.com/ibm/ado/commit/b66b736a5c7fca3dda88bc34c85ac2f4abad5d60)) - Michael Johnston
- (**core**) add script operation support for DiscoverySpace (#956) - ([8ae7897](https://github.com/ibm/ado/commit/8ae789741bbd7d91c8775273d6495ad3c46c68c8)) - Michael Johnston
- (**cplex-mip**) support warm start and improve memory handling (#1089) - ([0aa00a0](https://github.com/ibm/ado/commit/0aa00a0f3745ed75ba78114843bac0f5301d5226)) - Michael Johnston
- (**cplex-mip**) add custom experiment for solving MIP problems using CPLEX (#773) - ([204ee62](https://github.com/ibm/ado/commit/204ee62a4663a9d9f5db70544f1fef4d74f8d01e)) - Michael Johnston
- (**vllm-performance**) Install dependencies in benchmark execution worker (#952) - ([456d4e2](https://github.com/ibm/ado/commit/456d4e22c326d80f8fb04b955edbf21e194e7146)) - Christian Pinto
- (**vllm_performance**) add threadpool support for vLLM IOProcessors (#1039) - ([163eb88](https://github.com/ibm/ado/commit/163eb88ea43a205c3173a19bd2b07a165620cf7b)) - mgazz
- (**vllm_performance**) add OpenTelemetry traces support (#920) - ([bfa1fb2](https://github.com/ibm/ado/commit/bfa1fb2299c8207a858a716a93456936ca263e96)) - mgazz
#### Bug Fixes
- (**autoconf**) avoid_oom_recommender considers original number_gpus (#1019) - ([6d3cc8d](https://github.com/ibm/ado/commit/6d3cc8d1c4577676b5ed5cbfa2ff6c4e61bacb27)) - Vassilis Vassiliadis
- (**autoconf**) Error in mapping Llama3.1-8b (#946) - ([7545e24](https://github.com/ibm/ado/commit/7545e24c55f43f4233a242058d3fc1f566ffdbbc)) - Srikumar Venugopal
- (**cli**) avoid truncating df when related resources prevent deletion (#1174) - ([841578e](https://github.com/ibm/ado/commit/841578e5fbe6108cf76a410b5fd1378372a64538)) - Alessandro Pomponio
- (**core**) add support for scanning RECORD files for distributions on Python 3.10 (#1181) - ([b9e2865](https://github.com/ibm/ado/commit/b9e286563784fd0d318f6feb60e79b6632236650)) - Alessandro Pomponio
- (**core**) improve operation shutdown handling when SIGTERM is received (#977) - ([b5ed4f5](https://github.com/ibm/ado/commit/b5ed4f5ace82404cb622defca3b9f9f035b54607)) - Michael Johnston
- (**docs**) wrong fields and field values (#909) - ([840002d](https://github.com/ibm/ado/commit/840002d1840db7fa4ffa93715beb74e19df96492)) - Michael Johnston
- (**test**) dispose of mysql engines after use (#1080) - ([8bb0029](https://github.com/ibm/ado/commit/8bb0029756876a80fcdf94eca262fc3cafa98aff)) - Alessandro Pomponio
- (**test**) extend drain_queue timeout to prevent flaky results (#1056) - ([97b1f86](https://github.com/ibm/ado/commit/97b1f86118fb93a537c0c40c700fa9164815d941)) - Alessandro Pomponio
- (**test**) change trim parameters to prevent ci flakiness (#1053) - ([3c55ae4](https://github.com/ibm/ado/commit/3c55ae459a41eb10705211bfbe479763c7168f4d)) - Daniele Lotito
- (**tests**) ensure cloudpickle is installed in nonlocked tests (#1155) - ([b266665](https://github.com/ibm/ado/commit/b26666593b93739347a453cc3373ab9ac01c7387)) - Alessandro Pomponio
- (**vllm-performance**) propagate max_batch_tokens, gpu_memory_utilization, dtype, cpu_offload, and max_num_seq when creating a deployment (#1182) - ([dc8b978](https://github.com/ibm/ado/commit/dc8b978de49897e7f0c4f229f5e8f69e7fdf3a48)) - mgazz
- (**vllm-performance**) Delete deployment if timed out during creation (#1175) - ([3c37c06](https://github.com/ibm/ado/commit/3c37c06211ecd069ca4e25eddf768d25fb8cf258)) - Christian Pinto
- (**vllm_performance**) prevent TypeError when parsing renderer_num_workers (#1078) - ([ea82a16](https://github.com/ibm/ado/commit/ea82a168c42cab7ddbf83b7a4adb20dede58acbb)) - mgazz
- (**vllm_performance**) correct import paths in test_vllm_otlp_traces.py (#928) - ([48e51a1](https://github.com/ibm/ado/commit/48e51a13de85da71f8d415072c72e40150b4a5fe)) - mgazz
#### Performance Improvements
- (**test**) cap object store RAM usage for ray clusters in tests (#1055) - ([be73a5b](https://github.com/ibm/ado/commit/be73a5b630cc4e12ce8f100a2c5c46e298af10b0)) - Alessandro Pomponio
#### Documentation
- (**changelog**) add release notes for 1.8.0 (#911) - ([5454825](https://github.com/ibm/ado/commit/5454825a00c1de0142a02ddc68a0e669f79760b5)) - DRL-NextGen
- (**paper**) add citation file and badge for JOSS paper (#962) - ([a6f5ae1](https://github.com/ibm/ado/commit/a6f5ae16d1309dc799f9aa67a498ba036f7cbebb)) - Michael Johnston
- (**skills**) add skill to generate release notes (#1162) - ([30c1f05](https://github.com/ibm/ado/commit/30c1f0518c05490228c034bfa204c34475b61d4e)) - Alessandro Pomponio
- (**skills**) replace review slash command with skill (#1131) - ([9852b9b](https://github.com/ibm/ado/commit/9852b9b5e891a032e2c0c749f72c271beefe15dc)) - Alessandro Pomponio
- (**skills**) improve using-ado-cli skill score (#951) - ([3c46721](https://github.com/ibm/ado/commit/3c46721bf34f92a7b9433bb0fefc5650bc0428ad)) - yogesh-tessl
- (**skills**) add .agents (#922) - ([04a112e](https://github.com/ibm/ado/commit/04a112ea2c266950f86aff5eeedd1a9c0ac24834)) - Michael Johnston
- (**website**) update installation instructions (#1172) - ([d2915a0](https://github.com/ibm/ado/commit/d2915a07c9e8aae773fec8fa025bf25bd694475a)) - Alessandro Pomponio
- (**website**) update migration guide (#1166) - ([b7ea343](https://github.com/ibm/ado/commit/b7ea3431d3f4c25d3bf30fd13f20c345229d37e4)) - Alessandro Pomponio
- (**website**) publish migration guide on the website (#1133) - ([3a06fe1](https://github.com/ibm/ado/commit/3a06fe1dfda6e2a1360f39cdf77046f5eebd9343)) - Alessandro Pomponio
- (**website**) minor rewording (#1050) - ([47353b1](https://github.com/ibm/ado/commit/47353b1475f84eab51a77bc935ed089c5ecbd991)) - Michael Johnston
- (**website**) clarify Serve API is alpha and with no builtin security (#1040) - ([9508922](https://github.com/ibm/ado/commit/9508922b8f963efba615e0bf33f84817ef79eae6)) - Alessandro Pomponio
- (**website**) update local context example (#910) - ([6e8c83b](https://github.com/ibm/ado/commit/6e8c83b7eaad190475aca8a772b013ba2b3309ca)) - Alessandro Pomponio
- update README and website index page (#1171) - ([8a02c33](https://github.com/ibm/ado/commit/8a02c33aca731eeeca84fc71e0d9ed9702a11dab)) - Alessandro Pomponio
- update stale documentation (#1168) - ([5ed3fcc](https://github.com/ibm/ado/commit/5ed3fcc13251aed25433bd5e0e667fdfe180ca55)) - Alessandro Pomponio
- update versioning and plugin build backend instructions (#1165) - ([57c1b8c](https://github.com/ibm/ado/commit/57c1b8c5c92fee9390b04dabecd20ba9b877ba56)) - Alessandro Pomponio
- update roadmap and paper reference (#979) - ([f1179b9](https://github.com/ibm/ado/commit/f1179b9cd5891f0a5fc9fe991b81748ff41b59ee)) - Michael Johnston
- add note to issue template about editable installs (#915) - ([697df44](https://github.com/ibm/ado/commit/697df44f8344190885895bec9fe186c4f89016a1)) - Alessandro Pomponio
#### Tests
- print output in case of test failure (#1066) - ([75a074d](https://github.com/ibm/ado/commit/75a074d5739275bb1b9f199d198e90470675c1e9)) - Alessandro Pomponio
#### Build system
- (**anomalous-series**) decouple versioning from ado-core (#1141) - ([5cf6748](https://github.com/ibm/ado/commit/5cf674853e53b32effad7be518fd30aa92ecc242)) - Alessandro Pomponio
- (**autoconf**) decouple versioning from ado-core (#1151) - ([633c3c6](https://github.com/ibm/ado/commit/633c3c612dd2cbbbb1714aee70bcf95d9318bba3)) - Alessandro Pomponio
- (**autoconf**) restrict supported Python versions to <3.14 (#987) - ([5bc63c7](https://github.com/ibm/ado/commit/5bc63c754c1ed6d7c2205e218e88ced034ff110c)) - Alessandro Pomponio
- (**autoconf**) remove unused dependencies (#982) - ([bc9c500](https://github.com/ibm/ado/commit/bc9c500c095eb38f767fa6ca9ae182e34ad55487)) - Alessandro Pomponio
- (**ci**) update mend configuration file (#981) - ([3c5a3da](https://github.com/ibm/ado/commit/3c5a3da1f1873e647ca1dad45177b79fad830954)) - Alessandro Pomponio
- (**core**) remove no-priors workspace entry (#916) - ([3c77630](https://github.com/ibm/ado/commit/3c77630b8f336e7a23430d2d4e3313e10bd4ccd9)) - Alessandro Pomponio
- (**cplex-mip**) decouple versioning from ado-core (#1156) - ([26d50ce](https://github.com/ibm/ado/commit/26d50cee852863602c6c63bbcfcf8346870bbf2d)) - Alessandro Pomponio
- (**deps**) update dependencies (#1185) - ([b399f29](https://github.com/ibm/ado/commit/b399f29cbc435f0330b54b76a79268b90ffeac68)) - DRL-NextGen
- (**deps**) update dependencies (#1159) - ([d97ad1c](https://github.com/ibm/ado/commit/d97ad1cba9727a462c377303c8b324a452519e4b)) - DRL-NextGen
- (**deps**) update dependencies (#1128) - ([ff151b6](https://github.com/ibm/ado/commit/ff151b6d85a0f0d2c38933c72351efd044e80fae)) - DRL-NextGen
- (**deps**) update dependencies (#1099) - ([5abca97](https://github.com/ibm/ado/commit/5abca975fb48ced4f01bf66cac34ae151e07df32)) - DRL-NextGen
- (**deps**) update dependencies (#1057) - ([be48da2](https://github.com/ibm/ado/commit/be48da237c69019eeb0c09ccb0bd61691b30664f)) - DRL-NextGen
- (**deps**) update dependencies (#1036) - ([ab1decb](https://github.com/ibm/ado/commit/ab1decb00c988d62ed2f39341aab7c71b533c7b7)) - DRL-NextGen
- (**deps**) update dependencies (#1012) - ([094b610](https://github.com/ibm/ado/commit/094b610a4aad61d6bcb577326add18787ae92877)) - DRL-NextGen
- (**deps**) update dependencies (#997) - ([08402dd](https://github.com/ibm/ado/commit/08402ddd3a11554c4f24e19a1ac675757d48d1bb)) - DRL-NextGen
- (**deps**) update dependencies (#963) - ([e4f97f3](https://github.com/ibm/ado/commit/e4f97f3439c348ba61306052021ad846c4d4a41a)) - DRL-NextGen
- (**deps**) update dependencies (#953) - ([260a4ec](https://github.com/ibm/ado/commit/260a4ec29064a289a3fff7fd1355af820d2c0f2f)) - DRL-NextGen
- (**deps**) update dependencies (#943) - ([5d7cee8](https://github.com/ibm/ado/commit/5d7cee8cfe9ffede67205833ebd101ad977e1fd0)) - DRL-NextGen
- (**deps**) update dependencies (#907) - ([61f955d](https://github.com/ibm/ado/commit/61f955df5464a9c47a546ac8cc87675d3bd6c3a4)) - DRL-NextGen
- (**example-actuator**) decouple versioning from ado-core (#1157) - ([7200ba3](https://github.com/ibm/ado/commit/7200ba3fe89f0528d7322fd12c073060139c198a)) - Alessandro Pomponio
- (**hooks**) force only a subset of scopes for conventional commits (#1176) - ([dec3739](https://github.com/ibm/ado/commit/dec37394b15a2b80dda2c533d8c1ea8f08dadd8e)) - Alessandro Pomponio
- (**hooks**) update pre-commit hooks (#1160) - ([dbe0ce7](https://github.com/ibm/ado/commit/dbe0ce7204fa0cfa32e868744ea59bba558122d4)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#1100) - ([8571474](https://github.com/ibm/ado/commit/8571474ef4f3734692c59cdf41dbef9c1ee4aea5)) - DRL-NextGen
- (**hooks**) add new pre-commit hooks (#1069) - ([c54dd59](https://github.com/ibm/ado/commit/c54dd595bbbebea88bbe6ebc574b0dad5dbd18bc)) - Alessandro Pomponio
- (**hooks**) update pre-commit hooks (#1058) - ([d40c936](https://github.com/ibm/ado/commit/d40c93635ca0c6084a648aea60890df6f51df0d5)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#1037) - ([5dc4f18](https://github.com/ibm/ado/commit/5dc4f18563297720c0922e42ba710143fbe064c5)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#1013) - ([a703289](https://github.com/ibm/ado/commit/a70328916cec64f3b4ee6b31d67c7f5974ac97b4)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#986) - ([7e3fe9c](https://github.com/ibm/ado/commit/7e3fe9c87da03b23fe464df1fe3c29f3ce67d815)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#964) - ([86e8d6c](https://github.com/ibm/ado/commit/86e8d6c5623eaae967b5cd601529054ff06a7996)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#944) - ([b6b35a3](https://github.com/ibm/ado/commit/b6b35a3d6680387cb7dd3ba57835b1428b9b7bdc)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#934) - ([6ab1f5e](https://github.com/ibm/ado/commit/6ab1f5e1deb24feb118118216fffc6bc8130ff09)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#923) - ([8586f9a](https://github.com/ibm/ado/commit/8586f9ab8922966393e413bd62c55c92ca55985b)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#908) - ([a25f830](https://github.com/ibm/ado/commit/a25f83090fe28e144bda9c3065769a116ee0e639)) - DRL-NextGen
- (**plugins**) use Python entry points for loading actuators (#941) - ([af7f9df](https://github.com/ibm/ado/commit/af7f9df91ceb676f7ea51f43579f00b4a97f33f9)) - Alessandro Pomponio
- (**profile-space**) decouple versioning from ado-core (#1143) - ([8ac8cfb](https://github.com/ibm/ado/commit/8ac8cfb4ebe6b02a6a36286355a28c9448d2d82a)) - Alessandro Pomponio
- (**ray-tune**) decouple versioning from ado-core (#1139) - ([046e734](https://github.com/ibm/ado/commit/046e734ed6b576f3f313219dc5c90d4436469f97)) - Alessandro Pomponio
- (**ray-tune**) pin pandas to < 3 (#995) - ([b383c98](https://github.com/ibm/ado/commit/b383c9867e88fcde33af637c423ddb2c07886576)) - Alessandro Pomponio
- (**ray-tune**) remove python version-specific dep requirements (#983) - ([b3f927a](https://github.com/ibm/ado/commit/b3f927a75da62c3a0e661782663d01d54365fd1a)) - Alessandro Pomponio
- (**sfttrainer**) decouple versioning from ado-core (#1148) - ([2bd6338](https://github.com/ibm/ado/commit/2bd6338f764510310f09e85d62e860b807336837)) - Alessandro Pomponio
- (**trim**) decouple versioning from ado-core (#1137) - ([909d8f2](https://github.com/ibm/ado/commit/909d8f2e325a7cf87b15daf92f8f0d28eb1bd390)) - Alessandro Pomponio
- (**vllm-performance**) decouple versioning from ado-core (#1146) - ([a7daad1](https://github.com/ibm/ado/commit/a7daad18a28dcf5622abe892b7d9feefbb491764)) - Alessandro Pomponio
- (**vllm_performance**) require opencv-python-headless<5  (#1129) - ([c7bc873](https://github.com/ibm/ado/commit/c7bc87333ece9b4ad317ee628a74dfe9c6834ca9)) - Alessandro Pomponio
- handle forward slashes in git tags (#1153) - ([fd89ea0](https://github.com/ibm/ado/commit/fd89ea09c4b795143664b9404525b7c45737c4d2)) - Alessandro Pomponio
- update pinned python version to 3.11 (#994) - ([04c69ab](https://github.com/ibm/ado/commit/04c69ab067efa873fd08bec6a696a6b83fd24f00)) - Alessandro Pomponio
- ensure uv forks in dependency resolution (#992) - ([fbbabd5](https://github.com/ibm/ado/commit/fbbabd5d3b7b43da3d6b583fe681e33435e045a2)) - Alessandro Pomponio
- add support for Python 3.14 (#966) - ([6471c47](https://github.com/ibm/ado/commit/6471c4711402386273306534ad04a1aafd8745f5)) - Alessandro Pomponio
- remove .cra configuration (#937) - ([aa315ad](https://github.com/ibm/ado/commit/aa315ad150947b7f4e2cd2e613156ba0090942b9)) - Alessandro Pomponio
- add required ci checks file (#918) - ([8be02cb](https://github.com/ibm/ado/commit/8be02cbab7577e350b43e618b456ccf403994253)) - Alessandro Pomponio
#### Refactoring
- (**avoid_oom_recommender**) skip GPUs that are fewer than original GPUs (#1025) - ([bf32e11](https://github.com/ibm/ado/commit/bf32e11b93633ced12ab35c87bafe7053456912f)) - Vassilis Vassiliadis
- (**cli**) unify experiment id parsing across commands (#1125) - ([f359810](https://github.com/ibm/ado/commit/f3598109aff6edd6c2b6daac7772e6c5dd0303e5)) - Michael Johnston
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) rename --query to --filter (#1119) - ([e6c6e21](https://github.com/ibm/ado/commit/e6c6e2128c5892911a675b30b885325977a0030a)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) remove ado show details (#1115) - ([348449c](https://github.com/ibm/ado/commit/348449ce237e515f809f77b134d1d06c9f65edae)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) remove ado show summary (#1109) - ([b0fee02](https://github.com/ibm/ado/commit/b0fee021c25d7dd793b2e04c4e1e71747a16f650)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) remove legacy migrator system in ado upgrade (#1096) - ([e520937](https://github.com/ibm/ado/commit/e520937b901ea0e58aed319db5e29abd4085cf3d)) - Alessandro Pomponio
- (**cli**) improve exception handling in ado delete (#1092) - ([a2f9100](https://github.com/ibm/ado/commit/a2f9100ec91cac0f59a455fa919929800c8b4753)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) remove ado get measurementrequest (#1065) - ([9f5f848](https://github.com/ibm/ado/commit/9f5f84867fed16983b6b0dfb54286ca7c8ac1727)) - Alessandro Pomponio
- (**cli**) remove show requests and show results (#1049) - ([c6d972d](https://github.com/ibm/ado/commit/c6d972d7963027b6bc9cdab0d0a988374cf21321)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) rename ado show entities to ado show measurements (#1021) - ([6ec4c89](https://github.com/ibm/ado/commit/6ec4c89d7c1f04a24b6abeeb7b95e7d02125c7bd)) - Alessandro Pomponio
- (**cli**) do not rely on click for hidden plural and shorthand (#991) - ([01a817f](https://github.com/ibm/ado/commit/01a817f46ec7b8f479cd7c2448802b6b19a66e2d)) - Alessandro Pomponio
- (**cli**) remove ado template actuator (#976) - ([9f708d1](https://github.com/ibm/ado/commit/9f708d1b884f617ec9986b473bd1d2efab2d8aa0)) - Alessandro Pomponio
- (**cli**) --use-latest flag in show summary now behaves like other commands (#939) - ([1e39989](https://github.com/ibm/ado/commit/1e399891ebf26fc13033fa660b2743ea160f083f)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**core**) change import package from orchestrator to ado (#1179) - ([5687809](https://github.com/ibm/ado/commit/5687809f1c5cbb1c68842bfce3b6f46f6b46e38b)) - Alessandro Pomponio
- (**core**) rename dict_to_measurements to observed_property_values_from_dict (#1017) - ([9906ab6](https://github.com/ibm/ado/commit/9906ab63e3f938a2b654b4d8d049220283bf4b9d)) - Alessandro Pomponio
- (**core**) rename search operations to explore (#1122) - ([b472f66](https://github.com/ibm/ado/commit/b472f6664a5272c1da5d50114c97bf21b56500ae)) - Alessandro Pomponio
- (**core**) replace getRelatedResourceIdentifiers and getRelatedResources with get_resources_by_relationship (#1059) - ([606320a](https://github.com/ibm/ado/commit/606320aa33096d6162675ec2fd26b256a55eaa9c)) - Alessandro Pomponio
- (**core**) make simulate_json_contains_on_sqlite generic (#1031) - ([ab48f3f](https://github.com/ibm/ado/commit/ab48f3f2c58931fb58dd4fae435d961b3b4c13a6)) - Alessandro Pomponio
- (**core**) move sample store instantiation to class methods (#1027) - ([95b04b0](https://github.com/ibm/ado/commit/95b04b0f06f411d0fd8b9eb2fe60bcc02b7f8721)) - Alessandro Pomponio
- (**plugins**) rename configuration_model_default to example_configuration (#940) - ([c2d571f](https://github.com/ibm/ado/commit/c2d571f83d1d2700fb5b77b9cab0fa582c79ab7c)) - Alessandro Pomponio
- (**sfttrainer_download_hf_weights**) do not use ray (#1045) - ([05c1b5a](https://github.com/ibm/ado/commit/05c1b5aed737b45dfffe65d62ed8f5255db57b93)) - Vassilis Vassiliadis
- (**tests**) always override ado app dir in tests  (#1123) - ([400d815](https://github.com/ibm/ado/commit/400d8152ee6e546143e721fefd5f740440dca989)) - Michael Johnston
- (**vllm_performance**) expand parameter ranges for geospatial model benchmarking (#919) - ([77b2fad](https://github.com/ibm/ado/commit/77b2fad28f415bc69d7765347fa48d34c3d90f58)) - mgazz
#### Style
- use ruff format instead of black (#1169) - ([bbe6e5f](https://github.com/ibm/ado/commit/bbe6e5f24d6024de073f422194a299330ce0c7e7)) - Alessandro Pomponio

- - -

## [1.8.0](https://github.com/ibm/ado/compare/5720867a41f2e829b563070d9a81b50d17f7f687..1.8.0) - 2026-04-27
#### Features
- (**autoconf**) log confusion matrix (#743) - ([8bfc15e](https://github.com/ibm/ado/commit/8bfc15e0600e24df7eca89c3c377c7c257598ec5)) - Daniele Lotito
- (**cli**) add support for deleting multiple resources in ado delete (#885) - ([7f00b7a](https://github.com/ibm/ado/commit/7f00b7ae225e48a84631b378388b0b6b5dd4d381)) - Alessandro Pomponio
- (**cli**) add table output to show summary and change format names (#862) - ([0af35fe](https://github.com/ibm/ado/commit/0af35fe82e8bca49ee35ed368bf680c0dea35376)) - Alessandro Pomponio
- (**cli**) allow using --output-file with all output types  (#859) - ([78709a7](https://github.com/ibm/ado/commit/78709a749f25edfce0196db9bb5393b89b727da3)) - Alessandro Pomponio
- (**cli**) add --output-file flag to ado get (#838) - ([512cc11](https://github.com/ibm/ado/commit/512cc11ca2367be884771b6e03ffe2fb4eb371de)) - Alessandro Pomponio
- (**cli**) add request alias to measurementrequest (#830) - ([0eb6893](https://github.com/ibm/ado/commit/0eb6893398c951db2005a44450d274b82fe02bdf)) - Alessandro Pomponio
- (**cli**) add support for -o in ado contexts (#821) - ([cf1dbef](https://github.com/ibm/ado/commit/cf1dbef4a8ac69fbad1dfe3513fdc41739549a16)) - Alessandro Pomponio
- (**cli**) do not truncate id columns by default in ado get commands (#800) - ([7faf2bf](https://github.com/ibm/ado/commit/7faf2bf2f1fd402aa2eb3c079475c213b62920fb)) - Alessandro Pomponio
- (**cli**) add version info to ado get operators  (#785) - ([3015406](https://github.com/ibm/ado/commit/3015406c925d578f0f4e1c713799f3a1a5d256d7)) - Alessandro Pomponio
- (**cli**) add --no-trunc flag for commands that output rich tables (#797) - ([2d5ec00](https://github.com/ibm/ado/commit/2d5ec004ab4944699775515beafea11a1d1e3e94)) - Alessandro Pomponio
- (**cli**) add support for -o name in ado get  (#794) - ([924651a](https://github.com/ibm/ado/commit/924651a63105b930fa9795d85c199557f99f44b3)) - Alessandro Pomponio
- (**cli**) support legacy validators in ado upgrade (#629) - ([e12e2b4](https://github.com/ibm/ado/commit/e12e2b4185996cdd9515ac54a7cbcd430aab84d0)) - Alessandro Pomponio
- (**cli**) save latest operation id even on failure (#768) - ([310a25c](https://github.com/ibm/ado/commit/310a25ce75381934c0223bc037cabebcc0746343)) - Alessandro Pomponio
- (**core**) hash entity identifiers if too long for db engine (#846) - ([104ed79](https://github.com/ibm/ado/commit/104ed7969fde345465f3704bd04fec72e4ac6f30)) - Alessandro Pomponio
- (**core**) sort constitutive properties before generating entity id (#815) - ([9252032](https://github.com/ibm/ado/commit/9252032a68dde781ad5a0f18730550f96afded72)) - Alessandro Pomponio
- (**core**) compress ValidMeasurementResults when serializing (#769) - ([aba9ab6](https://github.com/ibm/ado/commit/aba9ab6fa53abf12f00c2c71b76c90519cb9abad)) - Alessandro Pomponio
- (**utilities**) enable auto width detection for rich renderables (#858) - ([328f782](https://github.com/ibm/ado/commit/328f7821b199e20b270aa17301b12550b3a14d4d)) - Alessandro Pomponio
- (**vllm_performance**) promote geospatial experiments to stable and add endpoint experiment with custom dataset (#744) - ([73c9d31](https://github.com/ibm/ado/commit/73c9d3193710ebf98f5d018bc5e9c2aedcde5069)) - Christian Pinto
- (**vllm_performance**) Check the vLLM deployment name is RFC1123 compliant (#759) - ([81565fe](https://github.com/ibm/ado/commit/81565fe8c3b0bc5c86196fda5a1ada03fa795818)) - Christian Pinto
#### Bug Fixes
- (**cli**) disable ray's uv run runtime env hook by default (#905) - ([502ba2b](https://github.com/ibm/ado/commit/502ba2b057e8895ee7386b18567dc92ce05e1564)) - Michael Johnston
- (**cli**) do not validate contexts in ado delete context (#841) - ([1963e96](https://github.com/ibm/ado/commit/1963e96790626e8c1de1946315ee790c5bdf49ef)) - Alessandro Pomponio
- (**cli**) ensure we find space ids in delete operation (#833) - ([0fdde08](https://github.com/ibm/ado/commit/0fdde08a3ec440d361dc10721bca88b0ff5722e9)) - Alessandro Pomponio
- (**cli**) prevent switching to invalid context and provide failsafe (#783) - ([0ec2eb8](https://github.com/ibm/ado/commit/0ec2eb8c096a8ffd98eaa739dbccf960dc09827d)) - Alessandro Pomponio
- (**cli**) use unique entities to calculate entities with no successful measurements (#782) - ([efdc084](https://github.com/ibm/ado/commit/efdc0842eb187efcae298ad9de8511d5b8112bcc)) - Alessandro Pomponio
- (**cli**) replace getResource with containsResourceWithIdentifier in delete operations (#748) - ([eeb6768](https://github.com/ibm/ado/commit/eeb67680392c1bdd02fe13390fcd962da2e4dea0)) - Alessandro Pomponio
- (**containers**) add Anaconda libs path to LD_LIBRARY_PATH (#760) - ([b9fa865](https://github.com/ibm/ado/commit/b9fa865ac90cde4daca93653ba71e5b9482efbe7)) - Christian Pinto
- (**core**) perform more thorough type checks in strip_binary_variable_types_data (#752) - ([f45fb8c](https://github.com/ibm/ado/commit/f45fb8c9bfe4975526e2b7e4b3682256c44e5f48)) - Alessandro Pomponio
- (**metastore**) correctly handle intermediate fields with underscores on sqlite (#900) - ([366ccd6](https://github.com/ibm/ado/commit/366ccd6a30faf0f04ecc4f3f90667154bf5ac512)) - Alessandro Pomponio
- (**operators**) prevent infinite recursion when calling non-explore operators (#886) - ([77e5b90](https://github.com/ibm/ado/commit/77e5b90b93c20c5c520e5b3765f8bf9f6fb1a4c0)) - Michael Johnston
- (**test**) include trim-custom-experiments in test configuration (#895) - ([b63156f](https://github.com/ibm/ado/commit/b63156fe9a3ace5ed1a9b85fea14e829a342d6a9)) - Alessandro Pomponio
- (**tests**) enable uv run pytest (#899) - ([dad2ace](https://github.com/ibm/ado/commit/dad2ace41e10e8b977a969b706a05a5ceb256d31)) - Michael Johnston
- (**vllm_performance**) do not override env if existing in deployment template (#896) - ([f1cc5dd](https://github.com/ibm/ado/commit/f1cc5dd3ed68798aa995d1232be2ddb16600e78e)) - Christian Pinto
#### Performance Improvements
- (**core**) use simpler checks for table existence (#670) - ([6113495](https://github.com/ibm/ado/commit/6113495422d7b84acca2d22067d78013a0052249)) - Michael Johnston
#### Documentation
- (**agents**) use updated output flags in skills (#865) - ([7893235](https://github.com/ibm/ado/commit/7893235d94cd22557cdb2b5cef5b07f337f59fa1)) - Michael Johnston
- (**agents**) add examining project skill (#852) - ([1aabbc0](https://github.com/ibm/ado/commit/1aabbc0e2230095ff6610e5a88829dde2f5a2652)) - Michael Johnston
- (**agents**) add note that domain range is closed on upper bound (#771) - ([1693461](https://github.com/ibm/ado/commit/169346135cef77241a77c70e49625b78585cd91a)) - Michael Johnston
- (**agents**) add conduct empirical study skill (#770) - ([9ae1340](https://github.com/ibm/ado/commit/9ae1340ca058667d296f90610dc0dbda06c2a200)) - Michael Johnston
- (**agents**) add operation and space examination skills (#754) - ([85feae9](https://github.com/ibm/ado/commit/85feae9c3d864beb6acada4f83c77df3b7746341)) - Michael Johnston
- (**changelog**) add release notes for 1.7.0 (#734) - ([5720867](https://github.com/ibm/ado/commit/5720867a41f2e829b563070d9a81b50d17f7f687)) - DRL-NextGen
- (**cli**) remove extra newlines from CLI docstrings (#883) - ([2bcff74](https://github.com/ibm/ado/commit/2bcff74b986cc147614348ea5fc96a0cf00fc8a5)) - Alessandro Pomponio
- (**contributing**) update tox environment name (#786) - ([4a6344b](https://github.com/ibm/ado/commit/4a6344b40d4523eaec41e56f14fbd9d840f45780)) - Alessandro Pomponio
- (**examples**) update TRIM example (#876) - ([8d7ce11](https://github.com/ibm/ado/commit/8d7ce1158e1e4445fc778ec148c30c68c87f8f7d)) - Alessandro Pomponio
- (**ordered_pip**) explain how to use the ordered-pip plugin (#747) - ([a8847e7](https://github.com/ibm/ado/commit/a8847e75a03f8f8d7735bf2738417f319c151828)) - Vassilis Vassiliadis
- (**sft_trainer**) recommend using ordered_pip (#753) - ([6797a1b](https://github.com/ibm/ado/commit/6797a1b3318fdf27d4d27447a4074aeb798e665f)) - Vassilis Vassiliadis
- (**sfttrainer**) update docs to point to ofed image (#764) - ([c84b00b](https://github.com/ibm/ado/commit/c84b00b51da83156cd044b524ba5bc8f014fff47)) - Vassilis Vassiliadis
- (**vllm_performance**) use optuna in vLLM performance endpoint example (#857) - ([aeb400f](https://github.com/ibm/ado/commit/aeb400fa773eaa52a2f97f9188210fbc953eed2b)) - Christian Pinto
- (**website**) how to guides (#870) - ([f234424](https://github.com/ibm/ado/commit/f234424b2cc115d7cfa20f3dbf62bb227e9c1fbe)) - Michael Johnston
- (**website**) add dedicated section to shorthands (#875) - ([ca2ee24](https://github.com/ibm/ado/commit/ca2ee243b1101d47d5f33c1d1be0bdfd9c6e6400)) - Alessandro Pomponio
- (**website**) group examples in navigation (#856) - ([f5c2233](https://github.com/ibm/ado/commit/f5c223354e8e69587a1451d480902de9908d90b7)) - Michael Johnston
- (**website**) fix and harmonize memoization docs (#849) - ([782ac65](https://github.com/ibm/ado/commit/782ac6535f302b2575405bf339263f40e69b2228)) - Michael Johnston
- (**website**) use correct yaml structure in properties-and-domains (#737) - ([a3ccacc](https://github.com/ibm/ado/commit/a3ccacc4a1a59872183244376692e111a1ed1380)) - Michael Johnston
- (**website**) refine core concepts docs (#727) - ([7db47c9](https://github.com/ibm/ado/commit/7db47c9e3050e673682cdd612f50a59765409c80)) - Michael Johnston
#### Tests
- (**core**) add tests for table_exists_query (#807) - ([74de1fc](https://github.com/ibm/ado/commit/74de1fc33c09ce31a588a325f43b32963546e228)) - Michael Johnston
#### Build system
- (**containers**) pass SOURCE_DATE_EPOCH as arg (#887) - ([bf4c09b](https://github.com/ibm/ado/commit/bf4c09bb31e427406b3598ac3032298926e63eec)) - Alessandro Pomponio
- (**containers**) remove the SFT dockerfile (#763) - ([41bb48b](https://github.com/ibm/ado/commit/41bb48b9932853245a59495944551aa2c27ec923)) - Vassilis Vassiliadis
- (**containers**) use explicit allowlist in .dockerignore (#762) - ([dcc9046](https://github.com/ibm/ado/commit/dcc9046ce2965e422b90a8a91ee0426e7b3aaeb0)) - Alessandro Pomponio
- (**core**) add entry point for random walk operator (#894) - ([39e0708](https://github.com/ibm/ado/commit/39e070850421bf1b4c76afc9f6656cc4642a2587)) - Michael Johnston
- (**deps**) update dependencies (#871) - ([25f7542](https://github.com/ibm/ado/commit/25f7542f6f5dcc58bbb9d655646ea5e44645c0a8)) - DRL-NextGen
- (**deps**) update dependencies (#831) - ([e9bb4bc](https://github.com/ibm/ado/commit/e9bb4bcfd899aa3059b48626aff48bbea60b4a06)) - DRL-NextGen
- (**deps**) update dependencies (#808) - ([b007c0c](https://github.com/ibm/ado/commit/b007c0c515a6a5dc58e2cbd6337a89459aa2d493)) - DRL-NextGen
- (**deps**) update dependencies (#755) - ([5518ab5](https://github.com/ibm/ado/commit/5518ab56f0dbfa9d3bf35c216c91042f0b536f0e)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#872) - ([d1e3664](https://github.com/ibm/ado/commit/d1e3664253c13b9ecf3a648076fd8ce03f810774)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#832) - ([8580c5f](https://github.com/ibm/ado/commit/8580c5f5188a0b4dbaf9ee2d420b031609ae81a7)) - DRL-NextGen
- (**hooks**) add default stage for hooks (#822) - ([6a85754](https://github.com/ibm/ado/commit/6a857545d198364546d9721fd97289a2dea17543)) - Alessandro Pomponio
- (**hooks**) update pre-commit hooks (#792) - ([24644be](https://github.com/ibm/ado/commit/24644be74f78937e4e94aec23b747d974cbe9eef)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#757) - ([3b2bf66](https://github.com/ibm/ado/commit/3b2bf66d91a42e39501ea4d7bd5ea2bb63cc805e)) - DRL-NextGen
- (**no-priors**) ensure package auto discovery emits no warnings (#740) - ([84ff8da](https://github.com/ibm/ado/commit/84ff8da16b75af4433bb16a0e6994322fc498ed0)) - Alessandro Pomponio
- (**trim**) use src layout (#741) - ([de2dbaf](https://github.com/ibm/ado/commit/de2dbaf0d9b1dfd061ec41b95f7a0ebd446a2cc3)) - Alessandro Pomponio
- (**vllm_performance**) require minimum datasets version (#775) - ([7b93270](https://github.com/ibm/ado/commit/7b93270208e0eaf4e8d1826ee8a7dea9747632b7)) - Christian Pinto
- (**vllm_performance**) Improved pypropject metadata (#761) - ([2c5ab5c](https://github.com/ibm/ado/commit/2c5ab5c33a4ebf92792c3d2f81f8e35fe5772665)) - Christian Pinto
- ensure artifacts use the same SOURCE_DATE_EPOCH (#882) - ([003e810](https://github.com/ibm/ado/commit/003e810dd7574670e14809409ad84e6144b44fd3)) - Alessandro Pomponio
- include timestamp in development wheels (#868) - ([e049b11](https://github.com/ibm/ado/commit/e049b1113a7c4cf49fc6df53e946ca57bb04e25e)) - Michael Johnston
- exclude uv.lock from tombi formatting (#795) - ([68a84ae](https://github.com/ibm/ado/commit/68a84ae9bdb01d2d31802bbd927252e8e17753c6)) - Alessandro Pomponio
#### Refactoring
- (**cli**) update df_to_output signature and add catch-all case (#879) - ([c169940](https://github.com/ibm/ado/commit/c1699400a590de659d893434470531fbf03ea55b)) - Alessandro Pomponio
- (**cli**) use --output-flag instead of --output/-o in ado template (#867) - ([259e3e1](https://github.com/ibm/ado/commit/259e3e1d1fadb92eddcfa6d671576192d89db02c)) - Alessandro Pomponio
- (**cli**) unify ado get handling for all resources (#854) - ([4cf88f1](https://github.com/ibm/ado/commit/4cf88f17b6e5a4010908efe26f103d417cd3b90a)) - Alessandro Pomponio
- (**cli**) update --use-latest to use db instead of local information (#837) - ([0e5e933](https://github.com/ibm/ado/commit/0e5e933f8c974752f36b2180934cee919687f19f)) - Alessandro Pomponio
- (**cli**) use table consistently for default output type in ado commands (#829) - ([8970efb](https://github.com/ibm/ado/commit/8970efb671dd010bc6a5034378b75d81d8e82fcb)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**cli**) use --output flag everywhere and output to stdout (#804) - ([ad26f6d](https://github.com/ibm/ado/commit/ad26f6d4d7674e1e148fceb31260a7f3c53cc073)) - Alessandro Pomponio
- (**cli**) simplify ado show summary space (#803) - ([43a469a](https://github.com/ibm/ado/commit/43a469aec04efb8541d0c77f82fd335425be82fd)) - Alessandro Pomponio
- (**core**) convert no-priors operator to a RandomWalk sampler (#877) - ([6f6703a](https://github.com/ibm/ado/commit/6f6703a1aeba3245b57a8e76ee3492a81eca497c)) - Daniele Lotito
- (**core**) remove support for actuators decorated with ray.remote (#878) - ([fec38a5](https://github.com/ibm/ado/commit/fec38a57c725d0aa4fabd1b4bc5d1ef05d1930d2)) - Alessandro Pomponio
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**core**) use class decorators for explore operators and add OperatorMetadata (#805) - ([2494c17](https://github.com/ibm/ado/commit/2494c17977914d6e9977f2540f420e8124552195)) - Michael Johnston
- (**core**) move dictionary utilities (#861) - ([ffab783](https://github.com/ibm/ado/commit/ffab783c6d3aecf6faba5cb7e767b0adcee9fbaf)) - Alessandro Pomponio
- (**core**) remove property type prefix in string representation (#780) - ([d55828a](https://github.com/ibm/ado/commit/d55828a165a58dbf3818464723d122978d29e10a)) - Christian Pinto
- (**core**) remove unused CoreResourceKinds (#767) - ([55f209e](https://github.com/ibm/ado/commit/55f209e4e44c1b98187449056dcd7e3b430065a0)) - Alessandro Pomponio
- (**metastore**) order resources returned by age (#904) - ([16288d0](https://github.com/ibm/ado/commit/16288d02b7acb23c9f325442a9d5fdedadfa9fd8)) - Alessandro Pomponio
#### Miscellaneous Chores
- (**agents**) add .bob link to .cursor (#745) - ([fbeaee7](https://github.com/ibm/ado/commit/fbeaee7aef85455b67c6c2469dd186ce977630ca)) - Alessandro Pomponio

- - -

## [1.7.0](https://github.com/ibm/ado/compare/651ded3949752b14349acf1b8dc7455eb03de305..1.7.0) - 2026-03-24
#### Features
- (**cli**) add space id to default ado get operation output (#694) - ([f90f8d4](https://github.com/ibm/ado/commit/f90f8d43224e05f2e2182a8270f5b68826fd53b9)) - Michael Johnston
- (**core**) add binary property domain to entity space rich representation (#714) - ([889c763](https://github.com/ibm/ado/commit/889c76332be4a99e50d80a7b65f84de7fbcfccf5)) - Alessandro Pomponio
- (**core**) enable aggregation of multi-valued metrics in ado show entities operation (#681) - ([a83fb3f](https://github.com/ibm/ado/commit/a83fb3f80c4a5b6cc12d80e6a7857a0f5c266204)) - Michael Johnston
- (**no-priors**) create standalone no-priors characterization operator (#686) - ([94c6b8f](https://github.com/ibm/ado/commit/94c6b8fb5987388640057b74eb924a3998dcbb8e)) - Daniele Lotito
- (**remote**) Support local .whl files in fromPyPI and validate source wheels (#702) - ([acf2868](https://github.com/ibm/ado/commit/acf286825ade639ca1e45753c27758dba0f4a425)) - Michael Johnston
- add generatorid parameter to SpacePoint.to_entity() (#724) - ([d753dee](https://github.com/ibm/ado/commit/d753deeaac1903464172ee329241a43e3fc08af5)) - Daniele Lotito
#### Bug Fixes
- (**core**) prevent invalid values in binary variable types (#718) - ([ec01c52](https://github.com/ibm/ado/commit/ec01c52162b9feb945491f9cac2e2d46e409d9a3)) - Alessandro Pomponio
- (**core**) validate DiscoveryOperationConfiguration parameters on creation  (#639) - ([d37708c](https://github.com/ibm/ado/commit/d37708c0f2d07b91e1a8467b9c4734c356376010)) - Alessandro Pomponio
- (**core**) check and restore db connections before executing statements (#661) - ([651ded3](https://github.com/ibm/ado/commit/651ded3949752b14349acf1b8dc7455eb03de305)) - Michael Johnston
- (**ray_tune**) use measurementspace to get virtual properties (#668) - ([32168d4](https://github.com/ibm/ado/commit/32168d4bbb31d7bcecc2c7d68e856e8059c164b4)) - Michael Johnston
- (**schema**) ensure experiment_id is string in series representation (#703) - ([443e8a5](https://github.com/ibm/ado/commit/443e8a586b59e5b7e3f1a576baa788a752f09276)) - Michael Johnston
- (**schema**) handle none values in aggregation (#662) - ([3d2c8d8](https://github.com/ibm/ado/commit/3d2c8d8db99ff3586f568ff76d928b78d810cd88)) - Michael Johnston
- (**vllm_performance**) Replace binary type with discrete type for properties in geospatial experiments (#720) - ([7d24eef](https://github.com/ibm/ado/commit/7d24eef9136b2130186d11de685634494ec16816)) - Christian Pinto
#### Performance Improvements
- (**core**) replace sqlalchemy.inspect with catalog checks for sql store table existence (#672) - ([8543c43](https://github.com/ibm/ado/commit/8543c4364db34ad590c1d3f344ef8cb523cbfcc4)) - Michael Johnston
- (**core**) add method to fetch multiple related resources from the DB at the same time (#682) - ([0986521](https://github.com/ibm/ado/commit/09865217de4cb753d425c2eb2f6c209dde9bc424)) - Michael Johnston
- (**core**) fetch missing entities in a batch when parsing measurement requests from db (#671) - ([03a65e7](https://github.com/ibm/ado/commit/03a65e7b3891159e571b26df85edb8a1d9e36a28)) - Michael Johnston
- (**core**) cache operation IDs belonging to space to avoid preflight DB lookup (#673) - ([fb82c5d](https://github.com/ibm/ado/commit/fb82c5dff31238b5ec8185883296a3e0d857c435)) - Michael Johnston
- (**core**) allow skipping loading experiment catalogs from DB when initializing DiscoverySpace (#675) - ([272fc81](https://github.com/ibm/ado/commit/272fc81079f20e4d137e249fc5a9c4d12c09d822)) - Michael Johnston
- (**core**) cache SQLAlchemy engine per database URL to eliminate redundant TCP handshake (#669) - ([ae68bf0](https://github.com/ibm/ado/commit/ae68bf0ef918e7bb55438b1840449b204f11f277)) - Michael Johnston
#### Documentation
- (**agents**) improve skill guidance (#719) - ([cffb68a](https://github.com/ibm/ado/commit/cffb68a0f3189115b19e1156c3d09769838394b9)) - Michael Johnston
- (**agents**) add instructions on using uv workspace packages when developing plugins (#708) - ([252c4ff](https://github.com/ibm/ado/commit/252c4fff155403c59523bef304d3592903cb4a35)) - Michael Johnston
- (**agents**) add instructions for writing skills (#707) - ([86640f1](https://github.com/ibm/ado/commit/86640f180970fee939d60d93526001bf8862b8a9)) - Michael Johnston
- (**agents**) improve rules for testing existing and new code (#705) - ([744bdda](https://github.com/ibm/ado/commit/744bdda5b6062721819e7f51612051b7b88aaa6c)) - Michael Johnston
- (**agents**) add skill for creating resource YAML (#680) - ([c46f3dc](https://github.com/ibm/ado/commit/c46f3dc0bb78000e28a82b9c3fd13ddf3386ccf7)) - Michael Johnston
- (**changelog**) add release notes for 1.6.0 (#660) - ([00c7de3](https://github.com/ibm/ado/commit/00c7de3e4ffe912badd942a61c6059182e7ee85e)) - DRL-NextGen
- (**cli**) clarify --output flag behavior for json/raw/yaml formats (#731) - ([1706698](https://github.com/ibm/ado/commit/1706698daf55f24f3ae6801a6c6a1eec70371640)) - Daniele Lotito
- (**sfttrainer**) fix typo in finetune-remotely.md (#667) - ([d8c9d98](https://github.com/ibm/ado/commit/d8c9d98f403ada8acb06b3123f7b92faea81625b)) - Vassilis Vassiliadis
- add prettier-ignore tags and format (#697) - ([cc2bca5](https://github.com/ibm/ado/commit/cc2bca581990bb2ffe2e6d76dab28d32be37f6dc)) - Alessandro Pomponio
#### Build system
- (**deps**) update dependencies (#728) - ([19d3201](https://github.com/ibm/ado/commit/19d3201357fcda25b7333ef79b382df35be50db9)) - DRL-NextGen
- (**deps**) update dependencies (#710) - ([77576ea](https://github.com/ibm/ado/commit/77576ea3b3cd6ddaf4cfadcabb2315d68ed2fdaa)) - DRL-NextGen
- (**deps**) update dependencies (#693) - ([c28385d](https://github.com/ibm/ado/commit/c28385d15efff5f417b2edef0db3e721344cb37b)) - Alessandro Pomponio
- (**deps**) update dependencies (#676) - ([9ad9d2a](https://github.com/ibm/ado/commit/9ad9d2a033c7f87794f8081c050ecc9c067bcb41)) - Alessandro Pomponio
- (**hooks**) update pre-commit hooks (#729) - ([0f0a66c](https://github.com/ibm/ado/commit/0f0a66c91590119761fcc227b3c8fedd385819aa)) - DRL-NextGen
- (**hooks**) update pre-commit hooks (#711) - ([c016992](https://github.com/ibm/ado/commit/c0169925699ee5d54387652760bf6e3e61121b2a)) - DRL-NextGen
#### Refactoring
- (**core**) remove ray.remote decorator from actuators (#698) - ([63bb898](https://github.com/ibm/ado/commit/63bb898ebacb2d353947f77929762d12f2c30693)) - Michael Johnston
- (**core**) remove orchestrator/plugins module (#695) - ([6ca67ea](https://github.com/ibm/ado/commit/6ca67ea9bfa8d5c66be74d36b2036cba4bfb795e)) - Michael Johnston

- - -

## [1.6.0](https://github.com/ibm/ado/compare/7061f8f399ea175fb3c2463d6813e0cb15f4d477..1.6.0) - 2026-03-05
#### Features
- (**agents**) add skill for remote execution (#633) - ([730cafb](https://github.com/ibm/ado/commit/730cafbe765987fef8ed6b4e1063003a1b48f9ce)) - Michael Johnston
- (**agents**) update ado cli skill (#632) - ([763ca03](https://github.com/ibm/ado/commit/763ca03737db7c1d0c4415f3face780850aeea71)) - Michael Johnston
- (**cli**) add --remote flag for automated remote Ray dispatch of ado CLI commands (#593) - ([493ea3f](https://github.com/ibm/ado/commit/493ea3f710f24e3a3aab2c3895ddcb2d51a62d06)) - Michael Johnston
- (**cli**) use rich dataframe print for default ado get handlers (#584) - ([cd302b9](https://github.com/ibm/ado/commit/cd302b9452f60bf9a3c0fc39c0b9b11c93d74c1c)) - Alessandro Pomponio
- (**cli**) use rich dataframe output in show commands (#586) - ([bcaa8df](https://github.com/ibm/ado/commit/bcaa8dfbd6def2a6e6233cc9e6e81355251f0fa0)) - Alessandro Pomponio
- (**cli**) show operator description when using --details (#577) - ([0c36592](https://github.com/ibm/ado/commit/0c3659211078416429794d1a28a9b7c154691147)) - Alessandro Pomponio
- (**cli**) use rich dataframe printing in ado get context (#571) - ([c75c33f](https://github.com/ibm/ado/commit/c75c33f750ca71d0c7c02209786e34d4b05120ec)) - Alessandro Pomponio
- (**cli**) use rich dataframe printing for ado get operators (#578) - ([ffc5000](https://github.com/ibm/ado/commit/ffc50009b84ae65fac554304254bf421dba421fb)) - Alessandro Pomponio
- (**core**) optionally raise exceptions when retrieving multiple resources (#626) - ([4643be5](https://github.com/ibm/ado/commit/4643be5b636c70e42dc39a65af3c7817bbcef24e)) - Alessandro Pomponio
- (**core**) optimize entity retrieval with targeted SQL queries (#498) - ([a218a8f](https://github.com/ibm/ado/commit/a218a8ff7eaa6414e07f4c73bee1ca114d8fece8)) - Michael Johnston
- (**trim**) introduce TRIM operator (#329) - ([b98c115](https://github.com/ibm/ado/commit/b98c11562c67dbb21fcd7e701cf7a732a62fc307)) - Daniele Lotito
- (**utilities**) add show_index parameter in dataframe_to_rich_table (#531) - ([4e3eb9f](https://github.com/ibm/ado/commit/4e3eb9f8ab6ef6fd4e28a0423fed41ca523aa399)) - Alessandro Pomponio
#### Bug Fixes
- (**actuators**) use correct typing and classmethod decorators  (#517) - ([3748122](https://github.com/ibm/ado/commit/374812283c4b9050a53a0ebca3223622f21a16b0)) - Michael Johnston
- (**autoconf**) retrain models due to dependency changes (#549) - ([f9ea285](https://github.com/ibm/ado/commit/f9ea285dca9cca49b6caddf463aa6568039c5f73)) - Daniele Lotito
- (**cli**) correctly handle virtual properties in show entities (#630) - ([66ade7e](https://github.com/ibm/ado/commit/66ade7eb4a4c6aab924febebf75ed5bf1cc6bfe7)) - Michael Johnston
- (**cli**) ensure run_experiment result is fully printed (#634) - ([4b63dbd](https://github.com/ibm/ado/commit/4b63dbddcd2e463908a7b1bf2bd45a3a968877a7)) - Michael Johnston
- (**core**) use raw dictionary instead of RuntimeEnv to initialise ray (#546) - ([a9a1470](https://github.com/ibm/ado/commit/a9a147085918397bf3c951f223684840486c1cb0)) - Michael Johnston
- (**custom_experiments**) compute measurement status instead of using if/else logic (#658) - ([0518e89](https://github.com/ibm/ado/commit/0518e89e5650beaac96cfe0634606c5542ba2a05)) - Michael Johnston
- (**ordered_pip**) check whether venv directory already exists (#573) - ([5c41146](https://github.com/ibm/ado/commit/5c4114633156a6d7216d0626c7824597d8a77889)) - Vassilis Vassiliadis
- (**ray_tune**) do not modify search algorithm parameters in-place (#656) - ([43aa30e](https://github.com/ibm/ado/commit/43aa30ed030215038074387c4c0b43184c518fef)) - Alessandro Pomponio
- (**ray_tune**) prevent deadlock by polling for critical errors (#657) - ([241cfff](https://github.com/ibm/ado/commit/241cfff5635071025961d4bbfcc0958aad283f32)) - Michael Johnston
- (**ray_tune**) serialize nan as a string (#642) - ([bab609a](https://github.com/ibm/ado/commit/bab609a565de9eb50f573929f32033e119de45db)) - Alessandro Pomponio
- (**ray_tune**) validate points_to_evaluate option of samplers (#631) - ([e03c4d3](https://github.com/ibm/ado/commit/e03c4d3bf1249e967b0026fd5685a5944ae973e0)) - Michael Johnston
- (**rifferla**) return correct model default when registering operator (#651) - ([10b8fda](https://github.com/ibm/ado/commit/10b8fda66f3c977d18ca51eb238f835f5289a30c)) - Alessandro Pomponio
- (**sfttrainer**) check whether dataset path exists and it's a file (#574) - ([3c1615c](https://github.com/ibm/ado/commit/3c1615c93aa9b4d0b499d08da5a6993b4db5a342)) - Vassilis Vassiliadis
- (**trim**) return correct model default when registering operator (#649) - ([c179f8b](https://github.com/ibm/ado/commit/c179f8b3ed84d367c896349f298286eb36392d43)) - Alessandro Pomponio
- (**trim**) propagate actuator configuration ids to sub-operations (#583) - ([3ed26da](https://github.com/ibm/ado/commit/3ed26daf38397f29ba12a916aed451f6b48fc217)) - Daniele Lotito
- (**vllm_performance**) Added 50th percentile metric to vllm bench experiments (#529) - ([3bc30b6](https://github.com/ibm/ado/commit/3bc30b655705510d8af0cb4e8efe300d25c123d5)) - Christian Pinto
#### Documentation
- (**agents**) improve planning for formulating a discovery problem (#635) - ([6393abb](https://github.com/ibm/ado/commit/6393abb58e9c1ceb8470ce2b335cc3927bf26d7c)) - Michael Johnston
- (**agents**) add coding agent skills and rules (#483) - ([b4a4042](https://github.com/ibm/ado/commit/b4a40428ac550c1f0e9ec77896eef4c37c8a3bf3)) - Michael Johnston
- (**autoconf**) clarify model versioning rationale (#604) - ([d772e19](https://github.com/ibm/ado/commit/d772e197631ee484e283abfa52cd867b621891a7)) - Daniele Lotito
- (**changelog**) update changelog (#519) - ([7061f8f](https://github.com/ibm/ado/commit/7061f8f399ea175fb3c2463d6813e0cb15f4d477)) - Alessandro Pomponio
- (**contributing**) exclude venv folders from checks (#526) - ([e48a20c](https://github.com/ibm/ado/commit/e48a20ca9c64338edfb9a3eaadcab6ab6005e3cf)) - Alessandro Pomponio
- (**test**) update test instructions (#530) - ([d91ce62](https://github.com/ibm/ado/commit/d91ce6263b007192d3d2283ce7924357d4e3695e)) - Alessandro Pomponio
- (**trim**) clarify requirements on target observed property (#602) - ([a3aa3e5](https://github.com/ibm/ado/commit/a3aa3e5804634879fc3a9be45b04e24c4f034d52)) - Daniele Lotito
- (**trim**) add readme for PyPI (#621) - ([2f9a49f](https://github.com/ibm/ado/commit/2f9a49fe2680f335b9e97d3b130d217272199ca1)) - Daniele Lotito
- (**trim**) fix the inclusion of TRIM example_yamls files (#552) - ([8982a1b](https://github.com/ibm/ado/commit/8982a1bcd8f2e145db571acfaeea802000ace24f)) - Vassilis Vassiliadis
- update prettier instructions (#551) - ([552e0cf](https://github.com/ibm/ado/commit/552e0cfd0e7b760cf69354fe9c388a6aeac1b24d)) - Michael Johnston
#### Build system
- (**autoconf**) require pandas version <3.0.0 (#648) - ([d17eb67](https://github.com/ibm/ado/commit/d17eb675f7a3f04d1fbc6be048f40cc5ead10d26)) - Srikumar Venugopal
- (**deps**) update dependencies (#659) - ([f4f411f](https://github.com/ibm/ado/commit/f4f411fe18ec2b22b5920409067163ebbb5b7f36)) - Alessandro Pomponio
- (**deps**) update dependencies (#637) - ([8096303](https://github.com/ibm/ado/commit/80963037504c014c388b79d22534f076f219dd93)) - Alessandro Pomponio
- (**deps**) update dependencies (#609) - ([fe08fc5](https://github.com/ibm/ado/commit/fe08fc5610b1a515f2ce3e44c9c56cd378af467e)) - Alessandro Pomponio
- (**deps**) update dependencies (#596) - ([6278d1a](https://github.com/ibm/ado/commit/6278d1a4e5a93821411887356463a71f6a493114)) - Alessandro Pomponio
- (**deps**) update dependencies (#592) - ([49a4e39](https://github.com/ibm/ado/commit/49a4e3940dee3b3e4480945876466c68adf07e2c)) - Alessandro Pomponio
- (**deps**) update dependencies (#561) - ([76878ea](https://github.com/ibm/ado/commit/76878ea17e77933c3c02fe07ebe61b9f7d15e5b6)) - Alessandro Pomponio
- (**deps**) require typer>=0.22.0 (#560) - ([5e5a67d](https://github.com/ibm/ado/commit/5e5a67d14d96f30fa584af121ff12fdd60872043)) - Alessandro Pomponio
- (**deps**) update dependencies (#545) - ([514fec9](https://github.com/ibm/ado/commit/514fec96d85e3114e9412adcd95dc452e12bad65)) - Alessandro Pomponio
- (**deps**) update dependencies (#525) - ([716a353](https://github.com/ibm/ado/commit/716a353bf1f25423720d3606cdc89b53f5a41515)) - Alessandro Pomponio
- (**pre-commit**) update hooks (#653) - ([209de45](https://github.com/ibm/ado/commit/209de4550601c7cf60510ac261662db044e778bb)) - Alessandro Pomponio
- (**pre-commit**) update hooks (#597) - ([03f9f99](https://github.com/ibm/ado/commit/03f9f9907ecd1cd1e82940340de46a8245421a45)) - Alessandro Pomponio
- (**pre-commit**) update hooks (#539) - ([7a5e388](https://github.com/ibm/ado/commit/7a5e388e58ea202930d9d7c4eef6d97a283887e1)) - Alessandro Pomponio
- (**trim**) rename wheel to ado-trim (#566) - ([91fc67a](https://github.com/ibm/ado/commit/91fc67a7f8cc85fee572cec92146bfc08c1c9599)) - Alessandro Pomponio
- (**trim**) use uv for pyproject (#542) - ([950b451](https://github.com/ibm/ado/commit/950b451c4c61b2632afaa3ce372a87faaa008b87)) - Alessandro Pomponio
- (**vllm_performance**) make vLLM and GuideLLM optional dependencies (#540) - ([c4d6572](https://github.com/ibm/ado/commit/c4d6572144759b4d6e1de97cbabcf6dbef10c86e)) - Christian Pinto
#### Refactoring
- (**cli**) add validation message for operation yaml in create --dry-run (#538) - ([f49b17f](https://github.com/ibm/ado/commit/f49b17fffbc7c08126d8ac26d1c029c4a2853f25)) - Alessandro Pomponio
- (**cli**) update ado get actuators (#535) - ([8effadc](https://github.com/ibm/ado/commit/8effadce9c6702dbcc02e2c4998660c4ab204cd9)) - Alessandro Pomponio
- (**core**) decorate actuators with ray.remote dynamically (#544) - ([0bbe316](https://github.com/ibm/ado/commit/0bbe31671b82038a14cd12a177f31099b53b3892)) - Michael Johnston
- (**examples**) update CSVSampleStores with old keys (#611) - ([f6236ea](https://github.com/ibm/ado/commit/f6236eabcd8ef754e76db54ca4e04533658febf9)) - Alessandro Pomponio
- (**metastore**) use ->> operator to extract unquoted fields (#582) - ([a57ddd9](https://github.com/ibm/ado/commit/a57ddd9969d82f3378ce7b8e7bfcbbe4cb9e0023)) - Alessandro Pomponio
- (**metastore**) remove legacy non tz-aware date handling (#581) - ([6ed0811](https://github.com/ibm/ado/commit/6ed08115d9e4abebc04c588561eccec2a432062f)) - Alessandro Pomponio
- (**modules**) simplify docstring extraction logic (#565) - ([32234f6](https://github.com/ibm/ado/commit/32234f6507d8268d7e96a85d9d00cd2aa1d12dcf)) - Alessandro Pomponio
- (**tests**) fix flaky test_get_experiments_plural_alias (#640) - ([fee447a](https://github.com/ibm/ado/commit/fee447a48145c45a1ca1bbd45e804e5dbeddac17)) - Alessandro Pomponio
- (**trim**) use Annotated pattern and default_factory for pydantic (#548) - ([6f4a79c](https://github.com/ibm/ado/commit/6f4a79cfc14f6394415ce1ae9c4b19b6a973430a)) - Alessandro Pomponio
#### Style
- add tombi for formatting toml (#543) - ([d8c3f73](https://github.com/ibm/ado/commit/d8c3f732112fffa4dfee35bf864181bf4f3208a0)) - Alessandro Pomponio

- - -

## [1.5.0](https://github.com/ibm/ado/compare/1.4.1..1.5.0) - 2026-02-09
#### Features
- (**autoconf**) Introduce a new recommender for per_device_train_batch_size (#500) - ([1cead81](https://github.com/ibm/ado/commit/1cead8158187be3c45a7134d81a909493d7fb825)) - Srikumar Venugopal
- (**cli**) add support for ado get experiments (#497) - ([624536e](https://github.com/ibm/ado/commit/624536e786c0f8e65a03448041cd00c5b177ad05)) - Alessandro Pomponio
- (**core**) improve sql sample store initialisation and allow for efficient refresh (#481) - ([167ca37](https://github.com/ibm/ado/commit/167ca3757c3084281a0ff782477a80076cb1e3b9)) - Alessandro Pomponio
#### Bug Fixes
- (**cli**) validate experiments against entity space in create space dry run (#513) - ([8a3c800](https://github.com/ibm/ado/commit/8a3c800dc95bcdd870f06b4e14dcefd4d97537bf)) - Michael Johnston
- (**core**) copy dictionary to avoid race condition in DiscoverySpaceManager monitorUpdates (#509) - ([b2d1736](https://github.com/ibm/ado/commit/b2d1736ad493a11d1c47a80e732ed0ebfa234241)) - Michael Johnston
- (**core**) set ray working_dir=None for local workers to support uv run (#506) - ([fa4f99e](https://github.com/ibm/ado/commit/fa4f99ef3204481a82456931f309b4ff47c2e39d)) - Michael Johnston
- (**modules**) avoid displaying empty spinner panel (#494) - ([5fa1980](https://github.com/ibm/ado/commit/5fa198091d109a82b2e7c4034dc816af43568d6f)) - Alessandro Pomponio
#### Performance Improvements
- (**core**) reuse engine for sql resource and sample stores (#492) - ([8e57d38](https://github.com/ibm/ado/commit/8e57d3834c5606f23eba59145db445e9f0cb4ee6)) - Alessandro Pomponio
#### Documentation
- (**changelog**) update changelog (#518) - ([db580b9](https://github.com/ibm/ado/commit/db580b9d3dc7ae73e99876bc5388c933ad65a2df)) - Alessandro Pomponio
- (**sfttrainer**) example for checking the variability of fms-hf-tuning measurements (#473) - ([174fcb1](https://github.com/ibm/ado/commit/174fcb1d4b8396b765ebccb20792511e5936e381)) - Vassilis Vassiliadis
- add instructions on ensuring locked dependencies (#515) - ([e85405c](https://github.com/ibm/ado/commit/e85405cd8823c3a98a78c4310167ee5d62d9062f)) - Alessandro Pomponio
#### Build system
- (**autoconf**) update lockfile (#493) - ([e210cfa](https://github.com/ibm/ado/commit/e210cfa51860df14fbb56a5928545c1c05ae92fe)) - Alessandro Pomponio
- (**autoconf**) update to autogluon 1.5 (#470) - ([f793fa6](https://github.com/ibm/ado/commit/f793fa65ca2bbc504461824d84a32ffa62688338)) - Srikumar Venugopal
- (**deps**) update dependencies (#512) - ([e6cbb71](https://github.com/ibm/ado/commit/e6cbb71447fd098f5deaf885059e2c88378476d9)) - Alessandro Pomponio
- (**deps**) update dependencies (#490) - ([93ac58a](https://github.com/ibm/ado/commit/93ac58ad104e3716ded976c5f07525b666955375)) - Alessandro Pomponio
- (**ray_tune**) require minimum numba version (#504) - ([9b366d8](https://github.com/ibm/ado/commit/9b366d86dc342d057ec9dd4f4a1009ab4fb7bc19)) - Alessandro Pomponio
- add detect-secrets to dev dependencies (#484) - ([ce9eb9d](https://github.com/ibm/ado/commit/ce9eb9da82c29c9e4f04f6b42cde93f1178d70d0)) - Alessandro Pomponio
#### Refactoring
- (**test**) support locked and unlocked runners (#508) - ([8dd15e0](https://github.com/ibm/ado/commit/8dd15e07e1bff214bd30a74415e5532be6f93c93)) - Alessandro Pomponio
- (**vllm_performance**) rename image_secret to image_pull_secret_name (#514) - ([b9677d7](https://github.com/ibm/ado/commit/b9677d70fd735fa63b827fa7a972aa8e5aa08132)) - Alessandro Pomponio

## [1.4.1](https://github.com/ibm/ado/compare/1.4.0..1.4.1) - 2026-01-30
#### Features
- (**schema**) add function for generating entity id (#471) - ([5412418](https://github.com/ibm/ado/commit/5412418a94d7558da39850b37176d29ec93c2135)) - Michael Johnston
#### Documentation
- (**changelog**) update changelog (#467) - ([f3628d7](https://github.com/ibm/ado/commit/f3628d71e7f4bd625368cc279c7dd3f04ff8c397)) - Alessandro Pomponio
- (**vllm_performance**) update examples (#472) - ([7955ce6](https://github.com/ibm/ado/commit/7955ce6bea42c7adaaab2f66ab3c48ce53d61dd1)) - Michael Johnston
- (**website**) various fixes (#475) - ([e76c968](https://github.com/ibm/ado/commit/e76c9685a88d009b1c9d324d0b4eddbfd8e37079)) - Michael Johnston
#### Refactoring
- use rich instead of IPython's pretty (#474) - ([622a242](https://github.com/ibm/ado/commit/622a242550893ef91aa0be14a50b20b9c0f84e74)) - Alessandro Pomponio
- move imports into type checking sections (#461) - ([ed8d6ab](https://github.com/ibm/ado/commit/ed8d6abfb73fdd00696a82b1a091eca313c06b4d)) - Alessandro Pomponio

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

## [1.3.3](https://github.com/ibm/ado/compare/1.3.2..1.3.3) - 2026-01-8
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

# Update protocol

This document describes the protocol for upgrading the experiments of the
SFTTrainer actuator.

## Protocol

- On n fms-hf-tuning release we grab the dependencies from the image and store
  them
  - We always record the dependencies even if we do not plan on upgrading the
    experiments
- If the experiments have not been upgraded to the latest fms-hf-tuning in last
  8 weeks (we've skipped three release versions) they are upgraded
  - The following direct dependencies of fms_hf_tuning are checked for changes
    also and relevant details recorded with the experiment release nodes:
    `torch`, `transformers`, `accelerate`, `flash-attn`
- Any experiments that cannot be supported by the image required to run the
  latest version are deprecated
  - If the image does not need to be changed, no experiments should be
    deprecated
  - If the image does need to be changed it may not be able to support the
    software stack versions of previous releases - these experiments will have
    to be deprecated
- In certain circumstances a upgrade may happen out-of-cycle, for example
  - A version of fms-hf-tuning is released that is flagged has a major change
    that should be used for up-coming tests
  - There is a change in the dependencies which have major positive performance
    implications
  - There is a bug or major regression that requires upgrading experiment (even
    without fms-hf-tuning release)
- However, except in the case of a bug, all upgrades are frozen while an active
  experiment campaign is running
  - Any upgrades that have been requested while the campaign is running are
    stored as "pending"
  - When the campaign is finished we perform an upgrade.
    - This will be to most recent fms-hf-tuning version, or the next scheduled
      one if that is closer.
    - If more than one upgrade was requested while upgrades were frozen only the
      last needs to be done.

Details

- fms_hf_tuning image location: quay.io/modh/fms-hf-tuning:release
- pinned packages extraction method:
  `docker run --rm quay.io/modh/fms-hf-tuning:release pip freeze`
- location for storing pinned packages:
  `https://github.com/ibm/ado/tree/main/orchestrator_plugins/controller/ado_actuators/fms_hf_tuning/packages`
- fms_hf_tuning dependencies to be checked on upgrade: `torch`, `transformers`,
  `accelerate`, `flash-attn`

### Questions to answer

Tick the checkboxes after providing an answer.

- [ ] Check whether this release introduces changes to the sft_trainer.main()
      and parse APIs which are not backwards compatible with latest supported
      version of fms-hf-tuning in our actuator
  - [ ] Yes -> will add a new file under
        <https://github.com/IBM/ado/tree/main/plugins/actuators/sfttrainer/ado_actuators/sfttrainer/wrapper_fms_hf_tuning/tuning_versions>
  - [ ] No changes
- [ ] Does this release require a change in experiment versioning?
  - [ ] Yes
  - [ ] No
- [ ] stored the packages of v{VERSION} under
      <https://github.com/IBM/ado/tree/main/plugins/actuators/sfttrainer/ado_actuators/sfttrainer/packages>
- [ ] looked at the releases of torch, transformers, accelerate, flash-attn
  - ...
- [ ] decided whether there is a need to update the experiments to include this
      version
  - [ ] Yes: I am upgrading the experiments so that we have the option of
        running experiments using the latest version of fms-hf-tuning.
  - [ ] No
- If updated experiments
  - [ ] update the current experiments to include the version
  - [ ] document the current experiments that this version is available

<!-- markdownlint-disable-next-line no-emphasis-as-heading -->
**Done when**

- [ ] we apply our upgrade protocol and write down our actions and decisions

## Expectations and Assumptions

We have to track if these assumptions are valid. If not we have to change to
protocol

- Changes involving an image change will not happen frequently meaning we often
  can support a range of experiment versions. This is based on prior experience
- There will not often be a reason to upgrade at a period < 8 weeks. This allows
  us to match an experiment version to a set of models to be tested for the
  duration of time (2-6 weeks) it takes to test them, without multiple new
  versions coming in.
  - The implication is that the upgrades are essentially "frozen" while a
    defined batch of model tests and experiments are being run (which takes 2-6
    weeks to complete)
  - We have to manage our experiment schedule so it matches.

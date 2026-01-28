# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
from enum import Enum

# AP: avoid misspellings and make it easier to change things down the line

# Resources
_ACTUATOR_CONFIGURATION_SINGULAR = "actuatorconfiguration"
_ACTUATOR_SINGULAR = "actuator"
_CONTEXT_SINGULAR = "context"
_DATA_CONTAINER_SINGULAR = "datacontainer"
_DISCOVERY_SPACE_SINGULAR = "discoveryspace"
_EXPERIMENT_SINGULAR = "experiment"
_MEASUREMENT_REQUEST_SINGULAR = "measurementrequest"
_OPERATION_SINGULAR = "operation"
_OPERATOR_SINGULAR = "operator"
_SAMPLE_STORE_SINGULAR = "samplestore"

# Output formats
_CONFIG = "config"
_CONSOLE = "console"
_CSV = "csv"
_DEFAULT = "default"
_JSON = "json"
_MARKDOWN_SHORT = "md"
_RAW = "raw"
_TABLE = "table"
_YAML = "yaml"


#################### ado ####################
class AdoLoggingLevel(Enum):
    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


#################### ado create ####################
class AdoCreateSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    CONTEXT = _CONTEXT_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


class AdoCreateWithResourceSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


#################### ado delete ####################
class AdoDeleteSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    CONTEXT = _CONTEXT_SINGULAR
    DATA_CONTAINER = _DATA_CONTAINER_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


#################### ado describe ####################
class AdoDescribeSupportedResourceTypes(Enum):
    DATA_CONTAINER = _DATA_CONTAINER_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    EXPERIMENT = _EXPERIMENT_SINGULAR


#################### ado edit ####################
class AdoEditSupportedEditors(Enum):
    VIM = "vim"
    VI = "vi"
    NANO = "nano"


class AdoEditSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    DATA_CONTAINER = _DATA_CONTAINER_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


#################### ado get ####################
class AdoGetSupportedOutputFormats(Enum):
    CONFIG = _CONFIG
    DEFAULT = _DEFAULT
    JSON = _JSON
    RAW = _RAW
    YAML = _YAML


class AdoGetSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    ACTUATOR = _ACTUATOR_SINGULAR
    CONTEXT = _CONTEXT_SINGULAR
    DATA_CONTAINER = _DATA_CONTAINER_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    MEASUREMENT_REQUEST = _MEASUREMENT_REQUEST_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    OPERATOR = _OPERATOR_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


#################### ado show details ####################
class AdoShowDetailsSupportedResourceTypes(Enum):
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR


#################### ado show entities ####################
class AdoShowEntitiesSupportedEntityTypes(Enum):
    MEASURED = "measured"
    MATCHING = "matching"
    MISSING = "missing"
    UNMEASURED = "unmeasured"


class AdoShowEntitiesSupportedOutputFormats(Enum):
    CONSOLE = _CONSOLE
    CSV = _CSV
    JSON = _JSON


class AdoShowEntitiesSupportedPropertyFormats(Enum):
    OBSERVED = "observed"
    TARGET = "target"


class AdoShowEntitiesSupportedResourceTypes(Enum):
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR


#################### ado show related ####################
class AdoShowRelatedSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    DATA_CONTAINER = _DATA_CONTAINER_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


#################### ado show requests ####################
class AdoShowRequestsSupportedOutputFormats(Enum):
    CONSOLE = _CONSOLE
    CSV = _CSV
    JSON = _JSON


class AdoShowRequestsSupportedResourceTypes(Enum):
    OPERATION = _OPERATION_SINGULAR


#################### ado show results ####################
class AdoShowResultsSupportedOutputFormats(Enum):
    CONSOLE = _CONSOLE
    CSV = _CSV
    JSON = _JSON


class AdoShowResultsSupportedResourceTypes(Enum):
    OPERATION = _OPERATION_SINGULAR


#################### ado show summary ####################
class AdoShowSummarySupportedOutputFormats(enum.Enum):
    MARKDOWN = _MARKDOWN_SHORT
    TABLE = _TABLE
    CSV = _CSV


class AdoShowSummarySupportedResourceTypes(Enum):
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR


#################### ado template ####################
class AdoTemplateSupportedResourceTypes(Enum):
    ACTUATOR = _ACTUATOR_SINGULAR
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    CONTEXT = _CONTEXT_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR


#################### ado upgrade ####################
class AdoUpgradeSupportedResourceTypes(Enum):
    ACTUATOR_CONFIGURATION = _ACTUATOR_CONFIGURATION_SINGULAR
    DATA_CONTAINER = _DATA_CONTAINER_SINGULAR
    DISCOVERY_SPACE = _DISCOVERY_SPACE_SINGULAR
    OPERATION = _OPERATION_SINGULAR
    SAMPLE_STORE = _SAMPLE_STORE_SINGULAR

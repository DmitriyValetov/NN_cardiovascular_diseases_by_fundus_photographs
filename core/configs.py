import enum


class ModelType(enum.Enum):
    regr = 1
    clss = 2



model2resize2batchsize = {
    'densenet121': {
        256: 20,
    },
    'resnet18': {
        256: 20,
    },
    "efficientnet-b0": {
        256: 20,
        512: 20,
    },
    "efficientnet-b1": {
        256: 20,
        512: 20,
    },
    "efficientnet-b2": {
        256: 20,
        512: 20,
    },
    "efficientnet-b3": {
        256: 20,
        512: 20,
        1024: 5,
    }
}

tgt2class_num = {
    "age": 1,
    "bmi": 1,
    "cholesterol": 1,
    "SBP": 1,
    "GFR": 1,
    "AF": 2,
    "sex": 2,
    "smoking": 2,
    "HR": 3,
    "DR": 3,
    "DM": 2,
}

tgt2Modeltype = {
    "DR": ModelType.clss,
    "HR": ModelType.clss,
    "DM": ModelType.clss,
    "sex": ModelType.clss,
    "smoking": ModelType.clss,
    "AF": ModelType.clss,
    "age": ModelType.regr,
    "bmi": ModelType.regr,
    "cholesterol": ModelType.regr,
    "SBP": ModelType.regr,
    "GFR": ModelType.regr,
}
import torch
from enum import Enum, unique

@unique
class Units(str, Enum):
    """Enum containing all available bytes units."""

    __slots__ = ()
    AUTO = "auto"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    GIGABYTES = "GB"
    TERABYTES = "TB"
    NONE = "B"

CONVERSION_FACTORS = {
    Units.TERABYTES: 1024**4,
    Units.GIGABYTES: 1024**3,
    Units.MEGABYTES: 1024**2,
    Units.KILOBYTES: 1024,
    Units.NONE: 1,
}

BYTES_SIZE_FACTORS = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
}

class ModelInfo():

    def __init__(
        self,
        model: torch.nn.Module,
        data_type: torch.dtype,
    ) -> None:
        self.model = model
        self.data_type = data_type

    def get_model_parameters(self):
        self.total_params = {"Total": sum(p.numel() for p in self.model.parameters())}
        format_str = self.format_output_str(self.total_params)
        print(format_str)

    def get_layer_parameters(self):
        self.layer_params = {}
        for name, layer in self.model.named_modules():
            if hasattr(layer, 'parameters'):
                layer_params = sum(p.numel() for p in layer.parameters())
                self.layer_params[name] = layer_params
        format_str = self.format_output_str(self.layer_params)
        print(format_str)

    def convert_parameter_count(self, parameter_num: int, unit: Units = Units.AUTO):
        if unit == Units.AUTO:
            for candidate_unit in [Units.TERABYTES, Units.GIGABYTES, Units.MEGABYTES, Units.KILOBYTES]:
                if parameter_num >= CONVERSION_FACTORS[candidate_unit]:
                    converted_value = parameter_num / CONVERSION_FACTORS[candidate_unit]
                    return (round(converted_value, 2), candidate_unit)
            return (parameter_num, Units.NONE)
        else:
            factor = CONVERSION_FACTORS.get(unit, 1)
            converted_value = parameter_num / factor
            return (round(converted_value, 4), unit)
    
    def format_output_str(self, info_dict):
        output_str = "="*30 + "\n"
        for name, parameter_num in info_dict.items():
            converted_parameter_size, unit = self.convert_parameter_count(parameter_num)
            bytes_size = converted_parameter_size * BYTES_SIZE_FACTORS[self.data_type]
            output_str += f"[{name}]  parameter num: {parameter_num} |  memory size: {bytes_size} {unit.value}\n"
        return output_str
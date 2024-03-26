# type: ignore
import re
from pydantic import BaseModel
from typing import Type, Any, Dict, TypeVar


try:
    import xmltodict
    import xml.etree.ElementTree as ET
except ImportError:
    import warnings

    warnings.warn(
        "xmltodict and xml.etree.ElementTree modules not found. Please install them to proceed. `pip install xmltodict`",
        ImportWarning,
        stacklevel=2,
    )


T = TypeVar("T", bound=BaseModel)


def json_to_xml(model: Type[BaseModel]):
    """Takes a Pydantic model and returns XML format for Anthropic function calling."""
    model_dict = model.model_json_schema()

    root = ET.Element("tool_description")
    tool_name = ET.SubElement(root, "tool_name")
    tool_name.text = model_dict.get("title", "Unknown")
    description = ET.SubElement(root, "description")
    description.text = (
        "This is the function that must be used to construct the response."
    )
    parameters = ET.SubElement(root, "parameters")
    references = model_dict.get("$defs", {})
    list_params = _add_params(parameters, model_dict, references)
    lambda_func = create_force_list_checker(list_params)
    


    # Why not include the name of all parameters that need List?
    if len(list_params) > 0:  # Need to append to system prompt for List type handling
        return (
            ET.tostring(root, encoding="unicode")
            + "\nFor any List[] types, include multiple <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME> tags for each item in the list. XML tags should only contain the name of the parameter."
        )
    else:
        return ET.tostring(root, encoding="unicode")



# This code is from Instructor, full credit to them. :)
def _add_params(root: ET.Element, model_dict: Dict[str, Any], references: Dict[str, Any], parent=None) -> List[Tuple[str, bool]]:  # Return value indiciates if we ever came across a param with type List
    # TODO: handling of nested params with the same name
    properties = model_dict.get("properties", {})
   
    #print(f"Current properties: {model_dict}")
    #print()
    
    nested_list_found = False
    
    list_params = []

    for field_name, details in properties.items():
        parameter = ET.SubElement(root, "parameter")
        name = ET.SubElement(parameter, "name")
        name.text = field_name
        type_element = ET.SubElement(parameter, "type")
        nested_model = None
        #print(field_name)
        #print(details)
        #print()

        # Get type
        if "anyOf" in details:  # Case where there can be multiple types
            # supports:
            # case 1: List type (example json: {'anyOf': [{'items': {'$ref': '#/$defs/PartialUser'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'title': 'Users'})
            # case 2: nested model (example json: {'anyOf': [{'$ref': '#/$defs/PartialDate'}, {'type': 'null'}], 'default': {}})
            field_types = []
            for d in details["anyOf"]:
                field_type = "unknown"
                if 'type' in d:
                    field_type = d['type']
                if '$ref' in d:
                    field_type = d['$ref']
                    nested_model = d['$ref']

                
            field_type = " or ".join(field_types)
            #print(f"anyOf case: {field_type}")
        else:
            #print(f"Unknown case!")
            field_type = details.get(
                "type", "unknown"
            )  # Might be better to fail here if there is no type since pydantic models require types
        
        if "array" in field_type and "items" not in details:
            raise ValueError("Invalid array item.")

        # Check for nested List
        if "array" in field_type and "$ref" in details["items"]:
            type_element.text = f"List[{details['title']}]"
            
            nested_list_found = True 
            list_params.append((field_name, parent))    
        # Check for non-nested List
        elif "array" in field_type and "type" in details["items"]:
            type_element.text = f"List[{details['items']['type']}]"
            
            list_params.append((field_name, None))
        else:
            type_element.text = field_type

        param_description = ET.SubElement(parameter, "description")
        param_description.text = details.get("description", "")

        # Checking if there are nested params
        #print(f"Isinstance(details, dict): {isinstance(details, dict)}")
        #print(f"$ref in details: {'$ref' in details}")
        if (isinstance(details, dict) and ("$ref" in details or nested_model is not None)):
            if nested_model is not None:
                reference = _resolve_reference(references, nested_model)
            else:
                reference = _resolve_reference(references, details["$ref"])
            #print(f"Reference: {reference}")

            if "enum" in reference:
                type_element.text = reference["type"]
                enum_values = reference["enum"]
                values = ET.SubElement(parameter, "values")
                for value in enum_values:
                    value_element = ET.SubElement(values, "value")
                    value_element.text = value
                continue

            nested_params = ET.SubElement(parameter, "parameters")
            list_params.extend(_add_params(
                nested_params,
                reference,
                references,
                parent=field_name
            ))
        elif field_type == "array" and nested_list_found:  # Handling for List[] type
            nested_params = ET.SubElement(parameter, "parameters")
            
            list_params.extend(_add_params(
                nested_params,
                _resolve_reference(references, details["items"]["$ref"]),
                references,
                parent=field_name,
            ))


def _resolve_reference(references: Dict[str, Any], reference: str) -> Dict[str, Any]:
    parts = reference.split("/")[2:]  # Remove "#" and "$defs"
    for part in parts:
        references = references[part]
    return references


def extract_xml(content: str) -> str:  # Currently assumes 1 function call only
    """Extracts XML content in Anthropic's schema from a string."""
    pattern = r"<function_calls>.*?</function_calls>"
    matches = re.findall(pattern, content, re.DOTALL)
    return "".join(matches)


def xml_to_model(model: Type[T], xml_string: str) -> T:
    """Converts XML in Anthropic's schema to an instance of the provided class."""
    parsed_xml = xmltodict.parse(xml_string)
    model_dict = parsed_xml["function_calls"]["invoke"]["parameters"]
    return model(
        **model_dict
    )  # This sometimes fails if Anthropic's response hallucinates from the schema

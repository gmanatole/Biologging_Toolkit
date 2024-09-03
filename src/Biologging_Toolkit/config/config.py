from pathlib import Path
from collections import namedtuple
import importlib.util
import os

package_name = 'Biologging_Toolkit'
spec = importlib.util.find_spec(package_name)
package_root = os.path.dirname(spec.origin)

print(package_root)
__path = {
    "xml_data_columns": Path(package_root, "config", "d3sensordefs.csv"),
}

SES_PATH = namedtuple("path_list", __path.keys())(**__path)
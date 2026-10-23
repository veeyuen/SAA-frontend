"""Standalone regression checks: no Streamlit/GCP connection required."""
import ast
import json
import pathlib
import types
import re
import pandas as pd

ROOT=pathlib.Path(__file__).parent
nb=json.loads((ROOT/'SEA_YOUTH_PRODUCTION_v12(1).ipynb').read_text())
cell=''.join(nb['cells'][9]['source'])
source_tree=ast.parse(cell)
notebook_names=['_norm_text','_contains_distance','map_youth_event']
notebook_nodes=[n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name in notebook_names]
nbmod=types.ModuleType('notebook_mapper')
nbmod.__dict__.update({'pd':pd,'re':re})
exec(compile(ast.Module(body=notebook_nodes,type_ignores=[]), '<notebook mapper>', 'exec'),nbmod.__dict__)

functions_source=(ROOT/'functions_with_youth_mapper.py').read_text()
functions_tree=ast.parse(functions_source)
fnnodes=[n for n in functions_tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_youth_map_norm_text','_youth_map_contains_distance','map_youth_event']]
fnmod=types.ModuleType('functions_mapper')
fnmod.__dict__.update({'pd':pd,'re':re})
exec(compile(ast.Module(body=fnnodes,type_ignores=[]), '<functions mapper>', 'exec'), fnmod.__dict__)

cases=[
 ('Dash','100','100m'), ('Dash','200','200m'), ('Run','3000','3000m'),
 ('Run','1500','1500m'), ('Run','800','800m'),
 ('Hurdles','100','100m Hurdles'),('Hurdles','110','110m Hurdles'),('Hurdles','400','400m Hurdles'),
 ('Steeplechase','2000','2000m Steeplechase'),('Steeplechase','3000','3000m Steeplechase'),
 ('2000m S/C','','2000m Steeplechase'),('3000m SC','','3000m Steeplechase'),
 ('100 Metres','','100m'),('1000m','',''),('Relay','400',''),('Race Walk','3000',''),
 ('Shot Put','', 'Shot Put'),('Discus Throw','', 'Discus Throw'),('Javelin','', 'Javelin Throw'),
 ('Hammer Throw','', 'Hammer Throw'),('Long Jump','', 'Long Jump'),('Triple Jump','', 'Triple Jump'),
 ('High Jump','', 'High Jump'),('Pole Vault','', 'Pole Vault'),
]
for event,distance,expected in cases:
 row=pd.Series({'EVENT':event,'DISTANCE':distance})
 a=nbmod.map_youth_event(row)
 b=fnmod.map_youth_event(row)
 assert a == b == expected, (event,distance,a,b,expected)

# No changes outside expected sections in the Streamlit source.
base=(ROOT/'streamlit_BQ_25(1).py').read_text()
updated=(ROOT/'streamlit_BQ_26.py').read_text()
added_import=', map_youth_event'
assert updated.replace(added_import,'',1).replace("\n    # SEA Youth mapping is intentionally independent of the general\n    # international (adult/Open) mapping populated by fetch_data().\n    # Match the authoritative SEA_YOUTH_PRODUCTION_v12 notebook for both\n    # NSG and non-NSG rows, without modifying unrelated reports.\n    data['MAPPED_EVENT'] = data.apply(map_youth_event, axis=1)\n",'',1)==base
original_functions=(ROOT/'functions (4).py').read_text()
assert functions_source.startswith(original_functions)
assert ast.parse(updated) and ast.parse(functions_source)
print(f'PASS: {len(cases)} Youth event cases match notebook v12')
print('PASS: Streamlit change confined to Youth remapping and import')
print('PASS: Existing functions.py code preserved verbatim')
print('PASS: AST syntax for both updated files')

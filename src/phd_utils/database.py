import pymongo
from datetime import datetime
import pandas as pd
from . import rename_signal_ids_to_var_ids
from loguru import logger

def get_data(col, initial_datetime: datetime, final_datetime: datetime, vars:list = None) -> list:
    vars_to_export = {'_id':0, 'time':1}
    if vars: [vars_to_export.update({var:1}) for var in vars]
    
    cursor = col.find({'time':{'$lt':final_datetime, '$gt':initial_datetime}}, vars_to_export).sort('time', pymongo.ASCENDING)
    # return list(map(lambda x: list(x.values())))
    
    # return await cursor.to_list()
    # Apparently it's efficient: The cursor does not actually retrieve each document from the server individually; it gets documents efficiently in large batches.
    return [data for data in cursor]


def get_data_db(initial_date: datetime, final_date: datetime,
                config: dict, var_ids:list[str] = None, signal_ids:list[str] = None,
                db_url: str = 'mongodb://10.10.105.209:27017',
                db_name: str = 'librescada',
                collection_name: str = 'operation_data',
                rename_signals_to_var_ids:bool = True, include_time:bool = True) -> pd.DataFrame:
    
    db_client = pymongo.MongoClient(db_url)
    db = db_client[db_name]
    col_origin = db[collection_name]
    
    # TODO: Should be coherent, just variables
    supported_objects = [(None, 'signal_id'), ('variables', 'signal_id'), ('measurements', 'sensor_id'), ('inputs', 'input_id')]
    
    import_vars = []
    if var_ids is None and signal_ids is None:
        # By default, get all variables in configuration

        for object_name, signal_type in supported_objects:
            if object_name:
                try:
                    import_vars.extend( [config[object_name][var_id][signal_type] for var_id in config[object_name]] )
                except KeyError:
                    logger.debug(f'No {object_name}-{signal_type} combo in config')
                break
            else:
                import_vars.extend( [config[var_id][signal_type] for var_id in config] )
                break
                
    else:
        if var_ids is not None:
            # Get only the variables specified in var_ids
            for var_id in var_ids:
                for object_name, signal_type in supported_objects:
                    if object_name:
                        if var_id in config[object_name]:
                            import_vars.append( config[object_name][var_id][signal_type] )
                            break
                    else:
                        if var_id in config:
                            import_vars.append( config[var_id][signal_type] )
                            break
                    
        
        if signal_ids is not None:
            import_vars.extend(signal_ids)
           
    if include_time:
        import_vars.insert(0, 'time')
        
    logger.debug(f'({len(import_vars)} variables) Variables to import IDs: {import_vars}')
    
    data = get_data(col_origin, initial_date, final_date, vars=import_vars)
    data = pd.DataFrame(data)
    
    logger.debug(f'Length of imported data {len(data)}. Imported variables ({len(data.columns)}): {list(data.columns)}')

    # Use sets to list variables that were not imported
    not_imported_vars = set(import_vars).difference(data.columns)
    if not_imported_vars:
        logger.warning(f'({len(not_imported_vars)}) Variables not imported: {not_imported_vars}')

    if 'time' in data.columns:
        data.set_index('time', inplace=True, drop=False)


    # Replace sensor ids with variable ids
    if rename_signals_to_var_ids:
        data = rename_signal_ids_to_var_ids(data, config)
            
    return data
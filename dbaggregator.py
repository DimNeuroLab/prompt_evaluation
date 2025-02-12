import sqlite3
import os
import time

def merge_table_from_db(conn, alias, table, config, mappings):
    """
    Process all rows from the attached DB's table.

    For each row:
      - Update foreign-key columns (if defined) using the parent table’s mapping.
      - Compute a unique key from the row (using the columns defined in config["unique"]).
      - If a row with that unique key already exists in the master, reuse its ID.
      - Otherwise, insert the row (using config["columns"]) into the master.
    Record a mapping from (alias, original_id) to the master ID.
    """
    cur = conn.cursor()
    # Get all rows from the attached database's table.
    query = f"SELECT * FROM {alias}.{table}"
    cur.execute(query)
    rows = cur.fetchall()

    for row in rows:
        row = dict(row)  # ensure we have a dictionary for ease of access
        old_id = row[config["pk"]]

        # Update foreign keys in this row (if any)
        for fk_col, parent_table in config.get("foreign_keys", {}).items():
            old_fk = row.get(fk_col)
            # (Assuming 0 or None means “no reference”)
            if old_fk is not None and old_fk != 0:
                # Look up the parent row’s master ID that was merged from this same attached DB.
                # (We recorded mappings using a key of (attached_db_alias, parent's old id).)
                if (alias, old_fk) in mappings[parent_table]:
                    row[fk_col] = mappings[parent_table][(alias, old_fk)]
                # If no mapping exists, you could alternatively query the master table
                # to try to find a matching parent row. For simplicity, we leave it as is.

        # Build the unique key tuple from the designated columns.
        unique_key = tuple(row[col] for col in config["unique"])

        # Query the master DB to see if a row with these unique values exists.
        where_clause = " AND ".join([f"{col} = ?" for col in config["unique"]])
        select_sql = f"SELECT {config['pk']} FROM {table} WHERE {where_clause}"
        cur.execute(select_sql, tuple(row[col] for col in config["unique"]))
        result = cur.fetchone()

        if result:
            master_id = result[0]
        else:
            # Insert the row into master using the columns defined in config["columns"]

            cols = config["columns"]
            insert_cols = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))

            insert_sql = f"INSERT INTO {table} ({insert_cols}) VALUES ({placeholders})"
            values = [row[col] for col in cols]
            print("inserting row", table, insert_cols, placeholders,values)
            cur.execute(insert_sql, values)
            master_id = cur.lastrowid

        # Record the mapping from the attached DB’s (alias, old_id) to the master’s ID.
        mappings[table][(alias, old_id)] = master_id
    cur.close()

def detach_with_retry(conn, alias, retries=5, delay=1):
    for attempt in range(retries):
        try:
            conn.execute(f"DETACH DATABASE {alias}")
            print(f"Detached {alias} successfully.")
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                print(f"Database {alias} is locked, retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(delay)
            else:
                raise
    raise Exception(f"Failed to detach {alias} after {retries} retries.")

def merge_attached_db(conn, db_path, merge_order, table_configs, mappings):
    """
    Attach a source DB (using an alias), process all tables in the given order,
    then detach the source DB.
    """
    # Create an alias based on the filename (without path and extension)
    alias = "db_" + os.path.splitext(os.path.basename(db_path))[0]
    print(f"Attaching database {db_path} as {alias} ...")
    conn.execute(f"ATTACH DATABASE '{db_path}' AS {alias}")

    for table in merge_order:
        print(f"  Merging table {table} from {alias} ...")
        config = table_configs[table]
        merge_table_from_db(conn, alias, table, config, mappings)
    conn.commit()

    # Optionally, run an integrity check before detaching.
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check")
    result = cur.fetchone()
    if result[0] != "ok":
        print(f"Integrity check for {alias} failed: {result[0]}")
    else:
        print(f"Integrity check for {alias} passed.")

    # Now try detaching, with a retry loop to overcome temporary locks.
    cur.close()
    detach_with_retry(conn, alias)
    print(f"Detached database {alias}.")





def merge_databases(master_db_path, attached_db_paths):
    """
    Merge multiple SQLite databases into the master database.

    Parameters:
      - master_db_path: path to the master SQLite database.
      - attached_db_paths: list of paths to other SQLite database files.
    """
    # Open the master database and set row_factory for dictionary access.
    conn = sqlite3.connect(master_db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")  # 30000 milliseconds = 30 seconds
    conn.execute("PRAGMA journal_mode = WAL")

    # This will hold per-table mappings: for each table,
    # mappings[table][(attached_db_alias, old_id)] = new_master_id.
    mappings = {table: {} for table in table_configs.keys()}

    # Define the merge order so that parent tables are processed before child tables.
    merge_order = ["Prompt", "Chat", "Message", "Setting", "Template"]

    for db_path in attached_db_paths:

        merge_attached_db(conn, db_path, merge_order, table_configs, mappings)
        conn.commit()

    conn.close()
    print("Merging complete.")


# === Configuration for the tables based on your schema ===

# (Here we define for each table:)
# - "pk": the primary key column.
# - "unique": list of columns whose combination defines uniqueness.
# - "columns": the columns to be inserted into the master (excluding the primary key,
#    which is auto-generated).
# - "foreign_keys": a dict mapping any foreign-key column to its parent table.
table_configs = {
    "Prompt": {
        "pk": "id",
        "unique": ["prefix", "suffix", "prompt", "endingMessage", "keepInitial", "botStart", "temperature"],
        "columns": ["createdAt", "prefix", "suffix", "prompt", "endingMessage", "keepInitial", "botStart",
                    "temperature"],
        "foreign_keys": {}
    },
    "Chat": {
        "pk": "id",
        "unique": ["pid", "studyId", "sessionId", "promptId", "name"],
        "columns": ["createdAt", "updatedAt", "name", "pid", "studyId", "sessionId", "promptId"],
        "foreign_keys": {"promptId": "Prompt"}
    },
    "Message": {
        "pk": "id",
        "unique": ["chatId", "position", "role", "text"],
        "columns": ["createdAt", "updatedAt", "role", "text", "position", "chatId"],
        "foreign_keys": {"chatId": "Chat"}
    },
    "Setting": {
        "pk": "id",
        "unique": ["openaiKey"],
        "columns": ["createdAt", "updatedAt", "openaiKey"],
        "foreign_keys": {}
    },
    "Template": {
        "pk": "id",
        "unique": ["name", "prompt"],
        "columns": ["createdAt", "updatedAt", "name", "prompt"],
        "foreign_keys": {}
    }
}

# === Example usage ===
if __name__ == "__main__":
    # # List the SQLite files you want to merge into the master.
    # # (These should have the same schema as the master.)
    # attached_db_paths = [
    #     "database2.sqlite",
    #     "database3.sqlite"
    #     # Add more file paths as needed.
    # ]
    #
    # # Path to your master database.
    # master_db_path = "master.sqlite"



    import glob
    # Run the merge.
    master_db_filename ="C:\\Users\\dimit\\Downloads\\sathyas_data_sqlite\\integrator-exp_db_4383.sqlite"
    db_dir="C:\\Users\\dimit\\Downloads\\sathyas_data_sqlite"
    db_names = glob.glob(db_dir+"\\*.sqlite")
    try:
        db_names.remove(master_db_filename)
    except ValueError as e:
        print(e)

    print("db_names:", db_names)


    merge_databases(master_db_filename, db_names)



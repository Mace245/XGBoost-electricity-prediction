import pandas as pd
from datetime import timedelta

def advance_date_range(input_file, output_file, days_to_advance):
    """
    Reads a CSV file, advances the dates in the 'DateTime' column by a specified
    number of days, and saves the result to a new CSV file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path where the output CSV file will be saved.
        days_to_advance (int): The number of days to advance the dates.
    """
    try:
        # Read the input CSV file into a pandas DataFrame.
        # The `parse_dates` argument tells pandas to automatically try and
        # convert the 'DateTime' column into proper date objects.
        print(f"Reading data from '{input_file}'...")
        df = pd.read_csv(input_file, parse_dates=['DateTime'])

        # Define the time difference to add.
        # In this case, it's 63 days to shift from 05-19 to 07-21.
        time_delta = timedelta(days=days_to_advance)

        # Add the time delta to every entry in the 'DateTime' column.
        # Pandas is smart enough to apply this operation to the entire column.
        print(f"Advancing dates by {days_to_advance} days...")
        df['DateTime'] = df['DateTime'] + time_delta

        # Save the modified DataFrame to a new CSV file.
        # `index=False` prevents pandas from writing the DataFrame index as a column.
        df.to_csv(output_file, index=False)
        print(f"Successfully created '{output_file}' with the advanced dates.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except KeyError:
        print(f"Error: The CSV file must contain a 'DateTime' column.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Script Execution ---
if __name__ == "__main__":
    # Define the input file name (your uploaded file)
    INPUT_CSV = 'test.csv'

    # Define the name for the new file that will be created
    OUTPUT_CSV = 'test_advanced.csv'

    # Set the number of days to shift the dates forward
    DAYS_TO_SHIFT = -7

    # Run the function to perform the date advancement
    advance_date_range(INPUT_CSV, OUTPUT_CSV, DAYS_TO_SHIFT)


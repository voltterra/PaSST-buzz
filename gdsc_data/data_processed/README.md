# Directory structure and 

This directory contain subdirectories with processed data.
The name of directory follows a certain naming convention that has a pattern:

`data_<processing_tag>/<processing_seq>_<processing_type>`

- `<processing_tag>`   - first 7 digits of sha1 taken from the output of unix date command. Created at the begining of a new processing scheme.
                       - Invoke `date | sha1` and take first 7 digits.
- `<processing_seq>`   - the order in which the processings were applied. 
                         For instance resample_1 would mean that the data was resampled from source, yielding "resample_1" as a name.
                         windowed_2 would be splitting data obtained on "resample_1" into windowed chunks.
- `<processsing_type>` - type of processing applied at a given <processing_seq>. Can be multiple entries (if processing was combined).

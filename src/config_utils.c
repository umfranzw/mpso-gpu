#include "config_utils.h"

cl_uint get_num_configs(FILE *config_file)
{
    //count the number of uncommented lines
    cl_uint count = 0;
    while (!feof(config_file))
    {
        skip_whitespace(config_file);

        if (!feof(config_file))
        {
            count++;

            //move past the valid line
            cl_int c = getc(config_file);
            while (c != EOF && c != '\n')
            {
                c = getc(config_file);
            }
        }
    }
    rewind(config_file);

    return count;
}

//returns 1 on success, 0 on error/eof
cl_int parse_param(FILE *file, char *patt, void *buf)
{
    skip_whitespace(file);
    int result = (int) fscanf(file, patt, buf);
    int success = (result != EOF && result > 0);
    if (success)
    {
        skip_whitespace(file);
    }

    return (cl_int) success;
}

void skip_whitespace(FILE *file)
{
    cl_int c = getc(file);

    //move to start of first line
    while (c != EOF && (c == ' ' || c == '\t' || c == '\n'))
    {
        c = getc(file);
    }
        
    //skip any commented lines
    while (c == '#')
    {
        //move to end of line
        while (c != EOF && c != '\n')
        {
            c = getc(file);
        }
        
        //move one char past the newline
        if (c != EOF)
        {
            c = getc(file);
        }

        //move to start of next line
        while (c != EOF && (c == ' ' || c == '\t' || c == '\n'))
        {
            c = getc(file);
        }
    }

    //if last char was valid, put it back
    if (c != EOF)
    {
        ungetc(c, file);
    }
}

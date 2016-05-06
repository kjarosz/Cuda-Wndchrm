#ifndef SIGNATURES_H
#define SIGNATURES_H



#include <string>
#include <vector>



#include "constants.h"



class Signatures
{
public:
	Signatures();
	~Signatures();

	void   add_signature(const char *sig_name, const char *filename, double value);

	double get_signature(const char *sig_name, const char *filename) const;
	double get_signature(const char *sig_name, int         row) const;
	double get_signature(int         col, const char *filename) const;
	double get_signature(int         col, int         row) const;

	// Remove functions at some point?

	int    get_signature_index(const char *name) const;
	int    get_filename_index(const char *name) const;

	void   get_sig_name(int col, char *output);
	void   get_file_name(int row, char *output);

	int    get_sig_count() const;
	int    get_file_count() const;

	void   clear();

	std::vector<std::string> get_sig_names() const;
	std::vector<std::string> get_filenames() const;

private:
	Signatures(const Signatures &other);
	Signatures &operator=(const Signatures &other);

	int find_in_array(char **arr, int len, const char *element) const;

	int insert_new_signature(const char *name);
	int insert_new_filename(const char *name);

	char **expand_array(char **arr, int len, int new_len);
	void expand_signature_container();
	void expand_filename_container();
	void expand_value_array(const int new_col_len, const int new_row_len);

	inline std::vector<std::string> get_array_copy(char **arr, int len) const;

private:

	// Size of the containers.
	int col_len;
	int row_len;

	// Number of actual values in the containers.
	int col_n;
	int row_n;

	char **sigs;    // Columns
	char **files;   // Rows
	double *values; // Data matrix
};



#endif // SIGNATURES_H
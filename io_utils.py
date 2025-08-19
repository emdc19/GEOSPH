__author__ = 'Alomir'

import os
import sys
import glob
import csv
import numpy as np
from pyevtk.hl import pointsToVTK

# PySPH imports
from pysph.solver.utils import load as pysph_load
parent_folder = '/Users/blair/Downloads/tunnel3d_DP'
output_folder = '/Users/blair/Downloads/tunnel3d_DP'

# =============================================================================
# ============================= INPUT METHODS =================================
# =============================================================================

def import_parts_data(csv_path=None):
    """ This method takes CSV file in 'CSV_PATH' reads its contents, and
    outputs them as a 2D NumPy array, where each row is a 1D array of doubles
    corresponding to the same row in the original file

    Parameters
    -----------
    :param csv_path: full path to the CSV file, f.e. '/usr/name/my_file.csv',
        of type string

    Output
    -----------
    :return: 2D NumPy array with the contents of the CSV file with type double

    Assumptions
    -----------
    Assumes the provided file in 'CSV_PATH' has a header, and that all all
    entries are numerical values
    """

    if os.path.exists(csv_path):
        try:
            return np.loadtxt(csv_path, delimiter=',', skiprows=1)

        except IOError:
            print("Could not read file: %s." % csv_path)
            print('Check if valid path and data is not corrupted.')
            sys.exit()

    raise Exception("Not a valid path!")


def import_simulation_parameters(txt_path=None):
    """ Given a TXT file in 'txt_path', read the contents of it and parse them
    for output

    Parameters
    -----------
    :param txt_path: full path to the TXT file, f.e. '/usr/name/my_file.txt',
        of type string

    Output
    -----------
    :return: returns a dictionary of the parsed rows in the TXT file

    Assumptions
    -----------
    Assumes the provided file in 'txt_path' is formatted <key>=<value>, with
    one entry as such per row of the file,
    starting at the first row
    """
    sim_params = {}

    try:
        file_data = open(txt_path)

        # Read a line at a time and store the information as a list of
        #  strings
        for line in file_data:
            # Removes the end of line character.
            line = line.rstrip('\n')

            # Creates a list with the key and respective value.
            line_data = line.split('=')
    
            # The first element (position '0')of the list is the key value
            key = line_data[0]

            # The second element (position '1') of the list is the value
            value = line_data[1]
            sim_params[key] = value

        file_data.close()
        return sim_params

    except IOError:
        print("Could not read file: %s." % txt_path)
        print('Check if valid path and data is not corrupted.')
        sys.exit()


def get_csv_header(csv_path=None):
    """ Reads the first row of a CSV file and returns the headers of each
    column as a list, in the same order as they appear in the original file

    Parameters
    -----------
    :param csv_path: full path to the CSV file, f.e. '/usr/name/my_file.csv',
        of type string

    Output
    -----------
    :return: returns a Python list of strings containing all the headers

    Assumptions
    -----------
    Assumes the provided file in 'CSV_PATH' has a header
    """

    if os.path.exists(csv_path):
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            return header

    raise Exception("Not a valid path!")


def separate_particle_data_arrays(particle_array, header_keys):
    """ This function takes a Numpy array where each column corresponds to a
    particle property, "particle_array", and a list of strings corresponding to
    each property. It splits 'particle_array' into n-individual arrays.

    Parameters
    -----------
    :param particle_array:  ndarray of size n-by-m and type np.double, with
        m = number of properties of a particle
    :param header_keys:  list of strings of size m

    Output
    -----------
    :return: Returns a dictionary of keys from header_keys to ndarrays from
        particle_arrays of size m

    Assumptions
    -----------
    Assumes there is one ndarray in particle_array and a corresponding header
    named 'type' in header_keys. To this particular field, the return type is
    changed to np.uintc
    """

    # Dictionary of keys to values with particles' properties
    part_data = {}

    # Read each array from "particle_array", assign it a key, and append it to
    #  the dictionary.
    if 'id' in header_keys:  # If property 'id' exists, change it to 'gid'
        header_keys[header_keys.index('id')] = 'gid'
    if 'mass' in header_keys:  # If property 'mass' exists, change it to 'm'
        header_keys[header_keys.index('mass')] = 'm'

    for i, key in enumerate(header_keys):
        part_data[key.lower()] = particle_array[:, i].astype(np.double)

    # This is based on the formatting of the CSV files used as input. Modify
    #  this if there are other non-double arrays.
    if 'gid' in part_data.keys():
        part_data['gid'] = particle_array[:, 0].astype(np.uintc)
    if 'type' in part_data.keys():
        part_data['type'] = particle_array[:, 1].astype(np.uintc)
    return part_data


# =============================================================================
# ============================= OUTPUT METHODS ================================
# =============================================================================

def _create_vtk_file(part_data, output_path, ftype=1, fname=None, version='S'):
    """ Method that writes pysph.ParticleArray information to a vtk file. This
    format is useful for analysis with ParaView, allowing for interpolation of
    data and beautiful graphs.

    Parameters
    -----------
    :param part_data: dictionary of pysph.base.particle_array.ParticleArray
    
        dumped by the main PySPH application
    :param output_path: path to the desired output directory
    :param ftype: type of input - 1 for my code, 0 for PySPH examples
    :param fname: desired name of the output file, f.e. "my_file"
    :param version: defines the level of detail in the output. Defaults to 'S'
        (for simplified) which tells the method to output the default
        ParticleArray 'fluid'. If version='F', outputs each ParticleArray to a
        separate file

    Output
    -----------
    :return: None - Output ParticleArrays to a VTK (VTU) file
    """
    step_data = {}  # Data to output to file

    # Used to retrieve the pertinent ParticleArray.
    if ftype == 1 and version.upper() == "S":
        part_types = ['sediment','boundary']
    else:
        part_types = part_data.keys()  # Set of array names: e.g. 'solid'

    # Get each ParticleArray type array
    for _type in part_types:
        type_arr = part_data[_type]  # The ndarray data for e.g. 'solid'

        # Dictionary of properties and values. 'all=False' makes it such that
        #  only properties setup for output are considered.
        disp_flag = False
        if ftype == 0:
            disp_flag = True
        props_dict = type_arr.get_property_arrays(all=disp_flag)

        if ftype == 1:
            extra_props = {}  # For manipulating strided properties

            # Break strided properties into multiple single properties
            sigma = props_dict['sigma']  # Stress
            sigma_props = {'sxx': sigma[::9], 'syy': sigma[4::9],
                           'szz': sigma[8::9], 'sxy': sigma[1::9],
                           'sxz': sigma[2::9], 'syz': sigma[5::9]}

            # Delete the original strided properties
            del props_dict['sigma']

            # Add sigma to the extra properties dictionary
            extra_props.update(sigma_props)

            # Special case for sediment particles
            if _type == 'sediment':
                disp = props_dict['disp']  # Accumulated displacement
                disp_props = {
                    '|u|': disp[::3],
                    '|v|': disp[1::3],
                    '|w|': disp[2::3],
                    '|Disp|': np.sqrt(
                        np.power(disp[::3], 2) + np.power(disp[1::3], 2) +
                        np.power(disp[2::3], 2)
                    )
                }

                # Total velocity
                vx = props_dict['u']
                vy = props_dict['v']
                vz = props_dict['w']
                v_prop = {
                    'vel': np.sqrt(
                        np.power(vx[::], 2) + np.power(vy[::], 2) +
                        np.power(vz[::], 2)
                    )
                }

                eps = props_dict['eps']  # Total strain tensor
                eps_e = props_dict['eps_e']  # Elastic strain tensor
                eps_p = props_dict['eps_p']  # Plastic strain tensor
                eps_dot = props_dict['eps_dot']  # Strain rate tensor
                eps_props = {'exx': eps[::9], 'eyy': eps[4::9],
                             'ezz': eps[8::9], 'exy': eps[1::9],
                             'exz': eps[2::9], 'eyz': eps[5::9],
                             'eexx': eps_e[::9], 'eeyy': eps_e[4::9],
                             'eezz': eps_e[8::9], 'eexy': eps_e[1::9],
                             'eexz': eps_e[2::9], 'eeyz': eps_e[5::9],
                             'epxx': eps_p[::9], 'epyy': eps_p[4::9],
                             'epzz': eps_p[8::9], 'epxy': eps_p[1::9],
                             'epxz': eps_p[2::9], 'epyz': eps_p[5::9],
                             'exx_dot': eps_dot[::9], 'eyy_dot': eps_dot[4::9],
                             'ezz_dot': eps_dot[8::9],
                             'exy_dot': eps_dot[1::9],
                             'exz_dot': eps_dot[2::9], 'eyz_dot': eps_dot[5::9]
                             }

                del props_dict['disp']
                del props_dict['eps']
                del props_dict['eps_e']
                del props_dict['eps_p']
                del props_dict['eps_dot']
                extra_props.update(disp_props)
                extra_props.update(v_prop)
                extra_props.update(eps_props)

            # Append the extra properties to the original dictionary
            props_dict.update(extra_props)

        # Keys to the properties
        props_keys = props_dict.keys()

        # For each property key in the type_arr, get the property values array
        #  and append to the key values.
        for key in props_keys:
            if key in step_data:
                np.ascontiguousarray(
                    np.concatenate((step_data[key], props_dict[key]))
                )
            else:
                step_data[key] = np.ascontiguousarray(props_dict[key])

    # Output the current ParticleArray to a VTK (VTU) file.
    path = os.path.join(output_path, fname)
    pointsToVTK(path, step_data['x'], step_data['y'], step_data['z'],
                step_data)

def _create_csv_file(part_data, output_path, ftype=1, fname=None, version='S'):
    """ Method that writes pysph.ParticleArray information to a csv file.

    Parameters
    -----------
    :param part_data: dictionary of pysph.base.particle_array.ParticleArray
    
        dumped by the main PySPH application
    :param output_path: path to the desired output directory
    :param ftype: type of input - 1 for my code, 0 for PySPH examples
    :param fname: desired name of the output file, f.e. "my_file"
    :param version: defines the level of detail in the output. Defaults to 'S'
        (for simplified) which tells the method to output the default
        ParticleArray 'fluid'. If version='F', outputs each ParticleArray to a
        separate file

    Output
    -----------
    :return: None - Output ParticleArrays to a csv file
    """

    # Used to retrieve the pertinent ParticleArray.
    if ftype == 1 and version.upper() == "S":
        part_types = ['sediment','boundary']
    else:
        part_types = part_data.keys()  # Set of array names: e.g. 'solid'

    # Get each ParticleArray type array
    for _type in part_types:
        step_data = {}  # Data to output to file
        type_arr = part_data[_type]  # The ndarray data for e.g. 'solid'

        # Dictionary of properties and values. 'all=False' makes it such that
        #  only properties setup for output are considered.
        disp_flag = False
        if ftype == 0:
            disp_flag = True
        props_dict = type_arr.get_property_arrays(all=disp_flag)

        if ftype == 1:
            extra_props = {}  # For manipulating strided properties

            # Break strided properties into multiple single properties
            sigma = props_dict['sigma']  # Stress
            sigma_props = {'sxx': sigma[::9], 'syy': sigma[4::9],
                           'szz': sigma[8::9], 'sxy': sigma[1::9],
                           'sxz': sigma[2::9], 'syz': sigma[5::9]}

            # Delete the original strided properties
            del props_dict['sigma']

            # Add sigma to the extra properties dictionary
            extra_props.update(sigma_props)

            # Special case for sediment particles
            if _type == 'sediment':
                disp = props_dict['disp']  # Accumulated displacement
                disp_props = {
                    '|u|': disp[::3],
                    '|v|': disp[1::3],
                    '|w|': disp[2::3],
                    '|Disp|': np.sqrt(
                        np.power(disp[::3], 2) + np.power(disp[1::3], 2) +
                        np.power(disp[2::3], 2)
                    )
                }

                # Total velocity
                vx = props_dict['u']
                vy = props_dict['v']
                vz = props_dict['w']
                v_prop = {
                    'vel': np.sqrt(
                        np.power(vx[::], 2) + np.power(vy[::], 2) +
                        np.power(vz[::], 2)
                    )
                }

                eps = props_dict['eps']  # Total strain tensor
                eps_e = props_dict['eps_e']  # Elastic strain tensor
                eps_p = props_dict['eps_p']  # Plastic strain tensor
                eps_dot = props_dict['eps_dot']  # Strain rate tensor
                eps_props = {'exx': eps[::9], 'eyy': eps[4::9],
                             'ezz': eps[8::9], 'exy': eps[1::9],
                             'exz': eps[2::9], 'eyz': eps[5::9],
                             'eexx': eps_e[::9], 'eeyy': eps_e[4::9],
                             'eezz': eps_e[8::9], 'eexy': eps_e[1::9],
                             'eexz': eps_e[2::9], 'eeyz': eps_e[5::9],
                             'epxx': eps_p[::9], 'epyy': eps_p[4::9],
                             'epzz': eps_p[8::9], 'epxy': eps_p[1::9],
                             'epxz': eps_p[2::9], 'epyz': eps_p[5::9],
                             'exx_dot': eps_dot[::9], 'eyy_dot': eps_dot[4::9],
                             'ezz_dot': eps_dot[8::9],
                             'exy_dot': eps_dot[1::9],
                             'exz_dot': eps_dot[2::9], 'eyz_dot': eps_dot[5::9]
                             }

                del props_dict['disp']
                del props_dict['eps']
                del props_dict['eps_e']
                del props_dict['eps_p']
                del props_dict['eps_dot']
                extra_props.update(disp_props)
                extra_props.update(v_prop)
                extra_props.update(eps_props)

            # Append the extra properties to the original dictionary
            props_dict.update(extra_props)

        # Keys to the properties
        props_keys = props_dict.keys()

        # For each property key in the type_arr, get the property values array
        #  and append to the key values.
        for key in props_keys:
            if key in step_data:
                np.ascontiguousarray(
                    np.concatenate((step_data[key], props_dict[key]))
                )
            else:
                step_data[key] = np.ascontiguousarray(props_dict[key])

        # Output the current ParticleArray to a csv file.
        path = os.path.join(output_path, _type + "_" +fname)
        with open(_type + "_" +fname+'.csv','w') as f:
            out = csv.writer(f, delimiter=',')
            out.writerow(step_data.keys())
            out.writerows(zip(*step_data.values()))



def convert_pysph_output(input_path, output_path, ftype,
                         version='S'):
    """
    Converts PySPH NPZ output to VTK or CSV files.

    This method takes the output files dumped from a PySPH application
    (either NPZ of HDF5 formats) located at 'input_path' directory, and convert
    them to either VTK or CSV formats. The converted files are saved in the
    directory with path 'output_path' with extension 'file_type'.

    Parameters
    -----------
    :param input_path: path to the NPZ or HDF5 result files from a PySPH
        application, f.e. ~path_to/<app>_output
    :param output_path: path to the directory where the converted files should
        be saved
    :param ftype: type of input - 1 for my code, 0 for PySPH examples
    :param file_type: extension of the files to be saved. Defaults to 'VTK',
        CSV also accepted
    :param version: defines the level of detail in the output. Defaults to 'S'
        (for simplified) which tells the method to output the default
        ParticleArray 'fluid'. If version='F', outputs each ParticleArray to a
        separate file

    Output
    -----------
    :return: None - Output ParticleArrays to either a VTK (VTU), or CSV
        extension files

    OBS
    -----------
    The capability of exporting CSV files is not implemented yet.
    """

    if os.path.exists(input_path) and os.path.exists(output_path):

        # Open directory.
        os.chdir(input_path)

        # Check if PySPH is of type NPZ or HDF5
        ext = "*.npz"
        if not glob.glob(ext):
            ext = "*.hdf5"

        # Run through all files with extension "ext" and process them.
        for file in glob.glob(ext):
            data = pysph_load(file)  # PySPH public interface method.
            particle_arrays = data['arrays']

            try:
                # if file_type.upper() == 'VTK':
                    # Format data and output it as a VTK file: file_name.vtk
                    _create_vtk_file(particle_arrays, output_path, ftype,
                                     os.path.splitext(file)[0], version)
                    _create_csv_file(particle_arrays, output_path, ftype,
                                     os.path.splitext(file)[0], version)
                # else:
                #     # Format data and output it as a CSV file: file_name.csv
                #     raise Exception("Method not implemented. Enter 'VTK' for "
                #                     "file type or leave default.")

            except IOError:
                print()
                print("Could not output file %s as a VTK file." % file)
                print('Check if valid path and data is not corrupted.')
                sys.exit()

# =============================================================================

if __name__ == '__main__':
    convert_pysph_output(parent_folder, output_folder, 1, version='S')
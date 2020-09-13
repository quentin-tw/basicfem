""" Reading gmsh file into basicfem input format """
import sys
import os
import ast
import numpy as np

class MshProcessor:
    def __init__(self, filename, show_obj_msg=True):
        self. show_obj_msg = show_obj_msg
        self.entity_dict = {}
        self.is_file_valid = False # used to do simple check of the file
        self._read_msh(filename)
        self.bc = []
        self.bcnh = []
        self.nf = []
        self.has_property = False
        self.bound_con = None
        self.bound_con_nonhomo = None
        self.properties = None
        self.materials = None
        

    def _read_msh(self, filename):
        node_summary = {"numEntityBlocks" : 0, 
                        "numNodes" : 0, 
                        "minNodeTag" : 0, 
                        "maxNodeTag" : 0}

        in_nodes_section = False
        in_nodes_summary = False
        in_node_entity = False
        in_node_tag = False
        in_node_coord = False
        in_node_tag = False
        current_entityDim = -1
        current_entity_ctr = 0
        entity_stop_ctr = 0
        # this info can later be used to assign the boundary condition
        entity_dict = {} 

        nodal_tag_list = []

        in_elements_section = False
        in_elements_summary = False
        in_elements_entity = False
        in_element_tag = False
        skip_elements = True
        element_ctr = 0
        element_ctr = 0
        element_list = []
        
        # check if the file contains $Node label and #Elements label
        status_check = [0, 0]
        self.is_file_valid = False


        for line in open(filename, 'r'):
            if line.strip() == "$Nodes":
                in_nodes_section = True
                in_nodes_summary = True
                status_check[0] = 1
                continue

            if line.strip() == "$EndNodes":
                in_nodes_section = False
                continue

            if in_nodes_section:
                if in_nodes_summary:
                    str_list = line.strip().split()
                    num_list = list(map(int, str_list))
                    (node_summary["numEntityBlocks"], node_summary["numNodes"],
                    node_summary["minNodeTag"], node_summary["maxNodeTag"]) \
                        = num_list
                    nodal_coord = np.zeros((node_summary["numNodes"], 2))
                    nodal_data_ctr = 0
                    in_node_entity = True
                    in_nodes_summary = False
                    continue
                
                if in_node_entity:
                    entity_info_list = list(map(int, line.strip().split()))
                    entity_dict[(entity_info_list[0], entity_info_list[1])] \
                        = []
                    entity_stop_ctr = entity_info_list[3] - 1
                    in_node_entity = False
                    in_node_tag = True
                    continue

                if in_node_tag:
                    nodal_tag_list.append(int(line.strip()))
                    entity_dict[(entity_info_list[0], entity_info_list[1])] \
                        .append(int(line.strip()))
                    if current_entity_ctr < entity_stop_ctr:
                        current_entity_ctr += 1
                    elif current_entity_ctr == entity_stop_ctr:
                        current_entity_ctr = 0
                        in_node_tag = False
                        in_node_coord = True
                    continue

                if in_node_coord:
                    nodal_coord[nodal_data_ctr, :] \
                        = list(map(float, line.strip().split()[0:2]))
                    nodal_data_ctr += 1
                    if current_entity_ctr < entity_stop_ctr:
                        current_entity_ctr += 1
                    elif current_entity_ctr == entity_stop_ctr:
                        current_entity_ctr = 0
                        in_node_coord = False
                        in_node_entity = True
                    continue

            if line.strip() == "$Elements":
                in_elements_section = True
                in_elements_summary = True
                status_check[1] = 1
                continue

            if line.strip() == "$EndElements":
                in_elements_section = False
                continue

            if in_elements_section:
                if in_elements_summary:
                    in_elements_entity = True
                    in_elements_summary = False
                    continue

                if in_elements_entity:
                    entity_info_list = list(map(int, line.strip().split()))
                    
                    # skipped all non 2D elements 
                    if entity_info_list[2] != 2 \
                            and entity_info_list[2] != 3:               
                        skip_elements = True
                    
                    element_ctr = 0
                    elements_in_block = entity_info_list[3]
                    in_elements_entity = False
                    in_element_tag = True

                    continue

                if in_element_tag:
                    if not skip_elements:
                        current_list = list(map(float, line.strip().split()))
                        current_list.append(1) # include property label
                        element_list \
                            .append(current_list)
                    element_ctr += 1

                    if element_ctr == elements_in_block:
                        in_element_tag = False
                        in_elements_entity = True
                        skip_elements = False
                        element_ctr = 0
                    continue
        
        if 0 not in status_check:
            self.is_file_valid = True
        # 2D only - ignored z coordinate data
        self.nodal_data = np.zeros((len(nodal_tag_list), 3))
        self.nodal_data[:,0] = np.asarray(nodal_tag_list)
        self.nodal_data[:,1:3] = np.asarray(nodal_coord)
        self.element_data = np.asarray(element_list)
        self.entity_dict = entity_dict

    def add_bc(self, method_label, input_list_label, 
               node_entity_tags, x_tag, y_tag):
        if input_list_label.lower() == 'bc':
            if x_tag not in [0,1] or y_tag not in [0,1]:
                print("Invalid x_tag or y_tag value - only 0 or 1 can be used.")
                return
            input_list = self.bc
        elif input_list_label.lower() == 'bcnh':
            input_list = self.bcnh
        elif input_list_label.lower() == 'nf':
            input_list = self.nf
        else:
            print("Invalid input list label. Use 'bc', 'bcnh', or 'nf'.")
            return



        if method_label == 'n': # add by list of node tags
            self._node_assignments(node_entity_tags, input_list, 
                                   x_tag, y_tag, 'bound_con')

        if method_label == 'e':  # add by entity tuples
            if type(node_entity_tags) == tuple and len(node_entity_tags) == 2:
                if node_entity_tags in self.entity_dict:
                    node_tags = self.entity_dict[node_entity_tags]
                    self._node_assignments(node_tags, input_list, x_tag, y_tag, 
                                        'bound_con')
                else:
                    print("Given entity tuple does not exists.")
                    return
            
            elif type(node_entity_tags) == list:
                for tuple_ in node_entity_tags:
                    node_tags = self.entity_dict[tuple_]
                    self._node_assignments(node_tags, input_list, x_tag, y_tag,
                                            'bound_con')

            else:
                print(node_entity_tags, type(node_entity_tags))
                print("invalid entity tuple(s) input.")

    def _node_assignments(self, node_tags, data_list, x_val, y_val, data_str):
        for node_tag in node_tags:
            if node_tag not in [item[0] for item in data_list]:
                data_list.append([node_tag, x_val, y_val])
            else:
                print("node ", node_tag, "is already added to the ", data_str)
                return

    def clear(self, data_str):
        if data_str == 'bc':
            self.bc = []
        elif data_str == 'bcnh':
            self.bcnh = []
        elif data_str  == 'nf':
            self.nf = []
    
    # currently for single property and single material
    def add_prop(self, thickness, youngs_modulus, density):
        self.has_property = True
        self.properties = np.array([[1, 1, thickness]])
        self.materials = np.array([[1, youngs_modulus, density]])
    
    def clear_prop(self):
        self.has_property = False
        self.properties = None
        self.materials = None
    
    def show_entities(self):
        print()
        for key, value in self.entity_dict.items():
            print('nodes in entity ',key,' ---')
            print(value)

    def show_prop(self):
        print('bound_con assignment state---')
        print(self.bc)
        print('bound_con_nonhomo state---')
        print(self.bcnh)
        print('nodal_forces state---')
        print(self.nf)
        print('properties state--')
        print(self.properties)
        print('materials state--')
        print(self.materials)

    def save_data(self, folder_name):
        self.bound_con = np.asarray(self.bc)
        self.bound_con_nonhomo = np.asarray(self.bcnh)
        self.nodal_forces = np.asarray(self.nf)


        cwd = os.getcwd()
        output_path = cwd + '/' + folder_name
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        np.savetxt(output_path + '/' +'nodal_data.txt', self.nodal_data)
        np.savetxt(output_path + '/' +'element_data.txt',self.element_data)
        if self.bound_con is not None:
            np.savetxt(output_path + '/' +'bound_con.txt', self.bound_con)
        if self.bound_con_nonhomo is not None:
            np.savetxt(output_path + '/' +'bound_con_nonhomo.txt', 
                       self.bound_con_nonhomo)
        if self.nodal_forces is not None:
            np.savetxt(output_path + '/' +'nodal_forces.txt', self.nodal_forces)
        if self.has_property:
            np.savetxt(output_path + '/' +'properties.txt', self.properties)
            np.savetxt(output_path + '/' +'materials.txt', self.materials)
        

                                                                              
def main():
    print('------------------------------------------------------------------')
    print('mshreader -- read msh file from gmsh and assign boundary data.')
    print('This script provides an interface to convert the msh file into')
    print('the input data format required by the basicfem.py. The nodal_data')
    print('and the element_data will be built once the msh file is read. The')
    print('bound_con, bound_con_nonhomo, nodal_forces, properties,',
           'and materials')
    print('will need to be assigned either by entity tuples or nodal labels.')
    print('------------------------------------------------------------------')
    test = 1
    has_exception = False
    is_file_valid = False
    filename = input('Enter the input msh file name: ')
    try:
        mesh = MshProcessor(filename)
        print('--------------------------------------------------------')
        print("The nodal_data and element_data matrices build complete.")
        print("Solver input data to be assigned.")
        print('--------------------------------------------------------')
        has_exception = False
        is_file_valid = mesh.is_file_valid
    except:
        has_exception = True    

    while not is_file_valid or has_exception:
        print('Invalid msh ile content. please try again.')
        filename = input('Enter the input msh file name: ')    
        try:
            mesh = MshProcessor(filename)
            print('--------------------------------------------------------')
            print("The nodal_data and element_data matrices build complete.")
            print("Solver input data to be assigned.")
            print('--------------------------------------------------------')
            has_exception = False
            is_file_valid = mesh.is_file_valid
        except:
            has_exception = True
        
        

    exit_flag = False

    while not exit_flag:
        print('\nOperations:')
        print('add   --> add boundary conditions, forces or material' +
              'properties')
        print('clear --> clear previously assigned data.')
        print('show  --> show the current state of mesh data.')
        print('showm --> show entity tuple - nodal groups mapping')
        print('save  --> save data into basicfem input format')
        print('exit  --> exit mshreader')
        option = input('Enter an operation' + 
                        '(add, clear, show, showm, save, and exit): ')
        option = option.strip()
        if option == 'add':
            option_add(mesh)
        elif option == 'show':
            mesh.show_prop()
        elif option == 'showm':
            mesh.show_entities()
        elif option == 'clear':
            option_clear(mesh)
        elif option == 'save':
            option_save(mesh)
        elif option == 'exit':
            exit_flag = True
        else:
            print("Invalid operation.")

def option_save(mesh):
    print("Specify a folder name in local directory. Data will be saved into") 
    print("this folder. If no such folder name exists, a new folder will be")
    print("created. If the folder exists, previous data will be overwritten.")
    folder_name = input("Enter a folder name: ")
    mesh.save_data(folder_name)
    print("Data saved.")

def option_clear(mesh):
    print("\nwhich data to clear? Enter one of the following label:\n" + 
          "`bc`  -> clear bound_con\n" + 
          "`bcnh` -> clear bounr_con_nonhomo\n" +
          "`nf`   -> clear nodal_forces\n" +
          "`prop` -> clear materials and properties")
    option = input("Enter a label: ")
    while option not in ['bc', 'bcnh', 'nf', 'prop']:
        print("Invalid label.")
        print("\nwhich data to clear? Enter one of the following label:\n" + 
        "`bc`  -> clear bound_con\n" + 
        "`bcnh` -> clear bounr_con_nonhomo\n" +
        "`nf`   -> clear nodal_forces\n" +
        "`prop` -> clear materials and properties")
    if option == 'prop':
        mesh.clear_prop()
    else:
        mesh.clear(option)
    
    print(option + " cleared.")

def option_add(mesh):
    print('\nWhat data you want to add? '+
           'Specify the label and then the input values.')
    add_list_input_descriptions()
    val_string = input('\nEnter label and corresponding values: ')
    add_args = val_string.split()
    add_args[1:] = [float(str_) for str_ in add_args[1:]]

    while not is_valid_add_args(add_args):
        add_list_input_descriptions()
        val_string = input('\nEnter label and corresponding values: ')
        add_args = val_string.split()
        add_args[1:] = [float(str_) for str_ in add_args[1:]]

    print('\nSpecify the nodal entities (a tuple) or nodal labels (a list).')
    print('The nodal labels are consistent to the label generates by the gmsh')
    print('software, so it might be helpful to look at the gmsh GUI and check')
    print('which nodal labels you want to specify.')
    print('Nodal entity are a mapping of entity tuples to nodal label lists,')
    print('and these nodal label groups are assigned by gmsh.')
    print()
    nodal_label_input_descriptions()
    label_str = input('\nEnter label and corresponding list: ')
    label_args = label_str.split(" ", 1)

    while not is_valid_label_args(label_args) or label_args[0] == 'showm' \
          or label_args[0] == 'show':
        if label_args[0] == 'showm':
            mesh.show_entities()
        elif label_args[0] == 'show':
            mesh.show_prop()
        
        nodal_label_input_descriptions()
        label_str = input('\nEnter label and corresponding list: ')
        label_args = label_str.split(" ", 1)


    label_args[1] = ast.literal_eval(label_args[1])
    
    if add_args[0] == 'prop':
        mesh.add_prop(add_args[1], add_args[2], add_args[3])
    else:
        mesh.add_bc(label_args[0], add_args[0], label_args[1], 
                     add_args[1], add_args[2])
    # add_bc(method_label, input_list_label, node_entity_tags, x_tag, y_tag)
    print('Assigned successfully.')

def is_valid_label_args(label_args):
    if label_args[0] not in ['e', 'n', 'show','showm'] :
        print("Invalid label. Need to be `e`, `n`, 'show' or `showm`.")
        return False
    
    if label_args[0] != 'showm' and label_args[0] != 'show':
        if not (label_args[1].startswith('(') or \
               label_args[1].startswith('[')):
            print("Invalid values. Must be a tuple or list of numbers/tuples.")
            return False  
        try:
            ast.literal_eval(label_args[1])
        except:
            print('Invalid expression of list.')
            return False
    
    return True

def add_list_input_descriptions():
    print('\nInput descriptions:')
    print('bc  <x_tag>  <y_tag> --> boundary conditions (bound_conn)')
    print('bcnh  <x_val>   <y_val> --> nonhomogeneous boundary conditions ' +
           '(bound_conn_nonhomo)')
    print('nf <x_val>  <y_val> --> nodal_forces')
    print('prop <thickness>  <youngs modulus>  <density> '+
           '--> properties and materials')

def nodal_label_input_descriptions():
    print('\nTo show the entity mapping, enter `showm`. To show current state')
    print('of added mesh data, enter `show`')
    print('Otherwise, enter `e` or `n` follows by a list of tuples or list of')
    print('nodal labels')
    print('Example:')
    print('')
    print('show  --> show the current state of mesh data.')
    print('showm  -->  show nodal entity - nodal label group mapping.')
    print('e [(0, 1), (1, 1)]  -->  add all the nodes in ' +
           'entity (0, 1) and (1, 1)')
    print('e (1,2)  -->  add all the nodes in entity (0, 1) and (1, 1)')
    print('n [3,4,5,6]  -->  add the nodal labels 3, 4, 5, and 6')


def is_valid_add_args(add_args):
    if add_args[0] not in ['bc', 'bcnh', 'nf', 'prop']:
        print('Invalid label.')
        return False
    
    if add_args[0] != 'prop' and len(add_args) != 3 :
        print('Incorrect amount of input arguments.')
        return False
    
    if add_args[0] == 'prop' and len(add_args) != 4 :
        print('Incorrect amount of input arguments.')
        return False 

    if add_args[0] == 'bc' and \
        (add_args[1] not in (0, 1) or add_args[2] not in (0, 1)):
        print('Invalid values for bc : 0 means free node and ' +
                                      '1 means fixed node.')
        return False
    
    return True

if __name__ == "__main__":
    main()


            



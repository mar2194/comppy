import numpy as np
import re
from pandas import *

def _isorient(val):
    """A function to test if a string representation of a ply orientation is in fact a ply orientation.

    Parameters
    ----------
    val : str
        A string representation of a number

    Returns
    -------
    bool : bool
        Was val a positive or negative integer?

    """
    try:
        if "+-" or "-+" in val:
            val = val.replace("+-", "").replace("-+", "")
        int(val)
        return True
    except ValueError:
        return False


class Layup:
    def __init__(self, layup_str=None, material_str=None, layup_list=None, material_list=None):
        self.layup_str = layup_str
        self.material_str = material_str

        if layup_str is not None and layup_list is None:
            self.layup_list = Layup.code_to_list(layup_str)
        else:
            self.layup_list = layup_list

        if material_str is not None and material_list is None:
            self.material_list = Layup.code_to_list(self.material_str, flag='material')
        else:
            self.material_list = material_list

    @staticmethod
    def _fix_pm(laminate_str, flag=None):
        """A method for apply +- operators to the laminate_str

        Parameters
        ----------
        laminate_str : str
            A partially-translated orientation or material code.

        flag : str
            A string indicating if the long_form_code being translated is a orientation or material code.
            Defaults to None indicating a orientation code.

        Returns
        -------
        laminate_str : str
            A partially-translated orientation or material code.

        """
        if flag is None or flag == 'orientation':
            regex_str = r"(\+\-\d{1,})"
        if flag == 'material':
            regex_str = r"(\+\-[a-zA-Z]{1,}\d*)"
        # iterator looking for all substrings of the type "+-45" or "+-MaterialName"
        find_uniq = re.finditer(regex_str, laminate_str)
        # the unique +- angles
        unique_angles = [y for x in set([i.group(0) for i in find_uniq]) for y in x.split("+-") if y != ""]
        for angle in unique_angles:
            laminate_str = laminate_str.replace("+-"+angle, angle+"/-"+angle)
        return laminate_str

    @staticmethod
    def _fix_mp(laminate_str, flag=None):
        """A method for apply -+ operators to the laminate_str

        Parameters
        ----------
        laminate_str : str
            A partially-translated orientation or material code.

        flag : str
            A string indicating if the long_form_code being translated is a orientation or material code.
            Defaults to None indicating a orientation code.

        Returns
        -------
        laminate_str : str
            A partially-translated orientation or material code.

        """
        if flag is None or flag == 'orientation':
            regex_str = r"(\-\+\d{1,})"
        if flag == 'material':
            regex_str = r"(\-\+[a-zA-Z]{1,}\d*)"
        # iterator looking for all substrings of the type "-+45" or "-+MaterialName"
        find_uniq = re.finditer(regex_str, laminate_str)
        # the unique +- angles
        unique_angles = [y for x in set([i.group(0) for i in find_uniq]) for y in x.split("-+") if y != ""]
        for angle in unique_angles:
            laminate_str = laminate_str.replace("-+"+angle, "-"+angle+"/"+angle)
        return laminate_str

    @staticmethod
    def _fix_sy_np(laminate_str, flag=None):
        """A method for applying repetition operators in the absence of parentheses.

        Parameters
        ----------
        laminate_str : str
            A partially-translated orientation or material code.

        flag : str
            A string indicating if the long_form_code being translated is a orientation or material code.
            Defaults to None indicating a orientation code.

        Returns
        -------
        laminate_str : str
            A partially-translated orientation or material code.

        """
        # Find all substrings of the type "45:2" or "MaterialName:2"
        if flag is None or flag == 'orientation':
            regex_strs = [r"(-\d+\:\d+)", r"(\d+\:\d+)"]
        if flag == 'material':
            regex_strs = [r"(-[a-zA-Z]{1,}\d*\:\d+)", r"([a-zA-Z]{1,}\d*\:\d+)"]
        for regex_str in regex_strs:
            find_uniq = re.findall(regex_str, laminate_str)
            # print(find_uniq)
            for uniq_str in find_uniq:
                # print('uniq str is:', uniq_str)
                num_repeats = int(uniq_str.split(":")[-1])
                angle = uniq_str.split(":")[0]
                # print("angle is:", angle)
                repeats = [angle for i in range(num_repeats)]
                repl_str = "/".join(repeats)
                laminate_str = laminate_str.replace(uniq_str, repl_str)
        return laminate_str

    @staticmethod
    def _fix_sy_pr(laminate_str):
        """A method for applying repetition operators outside of parentheses.

        Parameters
        ----------
        laminate_str : str
            A partially-translated orientation or material code.

        Returns
        -------
        laminate_str : str
            A partially-translated orientation or material code.

        """
        # Find all substrings of the type "(45/-45):2" or "(MaterialName/MaterialName):2"
        find_uniq = re.findall(r"(\([^\)\(]*\)\:\d+)", laminate_str)
        for uniq_str in find_uniq:
            num_repeats = int(uniq_str.split(":")[-1])
            sequence = uniq_str.split(":")[0].replace('(', '').replace(')', '')
            repeats = [sequence for i in range(num_repeats)]
            repl_str = "/".join(repeats)
            laminate_str = laminate_str.replace(uniq_str, repl_str)
            laminate_str = Layup._fix_sy_pr(laminate_str)
        return laminate_str

    @staticmethod
    def _fix_sy_ps(laminate_str):
        """A method for applying symmetry operators outside of parentheses.

        Parameters
        ----------
        laminate_str : str
            A partially-translated orientation or material code.

        Returns
        -------
        laminate_str : str
            A partially-translated orientation or material code.

        """
        # Find all substrings of the type "(45/-45):2s" or "(MaterialName/MaterialName):s"
        find_uniq = re.findall(r"(\([^\)\(]*\)\:\d*s)", laminate_str)
        for uniq_str in find_uniq:
            print(uniq_str)
            sequence = uniq_str.split(":")[0].replace('(', '').replace(')', '')
            repetitions = uniq_str.split(":")[1].replace('s', '')
            sequence_vals = sequence.split("/")
            if '\\' in(sequence_vals[-1]):
                mid_num = [sequence_vals[-1].replace('\\', '')]
                sequence_vals = sequence_vals[:-1]
                reversed_vals = sequence_vals[::-1]
                new_sequence = sequence_vals + mid_num + reversed_vals
            else:
                if repetitions != '':
                    repetitions = int(repetitions)
                    sequence_vals = sequence_vals*repetitions
                reversed_vals = sequence_vals[::-1]
                new_sequence = sequence_vals + reversed_vals
            repl_str = "/".join(new_sequence)
            laminate_str = laminate_str.replace(uniq_str, repl_str)
            laminate_str = Layup._fix_sy_pr(laminate_str)
            print(laminate_str)
        return laminate_str

    @staticmethod
    def _fix_sy_bs(laminate_str):
        """A method for applying symmetry operators outside of the brackets.

        Parameters
        ----------
        laminate_str : str
            A partially-translated shorthand laminate orientation code possibly containing a
            symmetry operator outside of the brackets.

        Returns
        -------
        laminate_str : str
            A long-hand laminate orientation code.

        """
        # Find all substrings of the type "[45/-45]:s" or "[MaterialName/MaterialName]:s"
        try:
            uniq_str = re.findall(r"(\[.*\]\:\d*s)", laminate_str)[0]
            sequence = uniq_str.split(":")[0].replace('[', '').replace(']', '')
            repetitions = uniq_str.split(":")[1].replace('s', '')
            sequence_vals = sequence.split("/")
            if '\\' in(sequence_vals[-1]):
                mid_num = [sequence_vals[-1].replace('\\', '')]
                sequence_vals = sequence_vals[:-1]
                reversed_vals = sequence_vals[::-1]
                new_sequence = sequence_vals + mid_num + reversed_vals
            else:
                if repetitions != '':
                    repetitions = int(repetitions)
                    sequence_vals = sequence_vals*repetitions
                reversed_vals = sequence_vals[::-1]
                new_sequence = sequence_vals + reversed_vals
            repl_str = "/".join(new_sequence)
            laminate_str = laminate_str.replace(uniq_str, repl_str)
            laminate_str = Layup._fix_sy_pr(laminate_str)
            laminate_str = "[" + laminate_str + "]"
            return laminate_str
        except IndexError:
            return laminate_str

    @staticmethod
    def trans_code_or(laminate_str):
        """A method for translating short-hand ply-orientation codes.

        Parameters
        ----------
        laminate_str: str
            String representing a short-hand ply orientation code.

        Returns
        -------
        laminate_str: str
            String representing the long-hand ply orientation code.

        """
        laminate_str = Layup._fix_pm(laminate_str)
        laminate_str = Layup._fix_mp(laminate_str)
        laminate_str = Layup._fix_sy_np(laminate_str)
        laminate_str = Layup._fix_sy_ps(laminate_str)
        laminate_str = Layup._fix_sy_pr(laminate_str)
        laminate_str = Layup._fix_sy_bs(laminate_str)

        return laminate_str

    @staticmethod
    def trans_code_ma(laminate_str):
        """A method for translating ply-orientation material codes.

        Parameters
        ----------
        laminate_str: str
            String representing a material orientation code.

        Returns
        -------
        laminate_str: str
            String representing the longhand code of materials.

        """
        laminate_str = Layup._fix_pm(laminate_str, flag='material')
        laminate_str = Layup._fix_mp(laminate_str, flag='material')
        laminate_str = Layup._fix_sy_np(laminate_str, flag='material')
        laminate_str = Layup._fix_sy_ps(laminate_str)
        laminate_str = Layup._fix_sy_pr(laminate_str)
        laminate_str = Layup._fix_sy_bs(laminate_str)

        return laminate_str

    @staticmethod
    def code_to_list(long_form_code, flag=None):
        """A method for returning lists of materials or orientations of a layup sequence.

        Parameters
        ----------
        long_form_code : str
            A string representing the long-form laminate layup orientation or material code.
        flag : str
            A string indicating if the long_form_code being translated is a orientation or material code.
            Defaults to None indicating a orientation code.

        Returns
        -------
        code_list : list
            A list of the individual elements in the layup.
            For orientations this is a list of floats.
            For materials this is a list of strings.

        """
        long_form_code = long_form_code.replace('[', '').replace(']', '')
        if flag is None or flag == 'orientation':
            code_list = [float(x) for x in long_form_code.split('/')]
        if flag == 'material':
            code_list = [x.replace('-', '') for x in long_form_code.split('/')]
        return code_list

    @classmethod
    def gen_from_short(cls, orientation_layup, material_layup):
        """Alternate constructor for generating Layup objects from short-hand codes.

        Parameters
        ----------
        orientation_layup : str
            Shorthand ply orientation code
        material_layup : str
            Shorthand material orientation code

        Returns
        -------
        An instance of the Layup class

        """
        layup_str = Layup.trans_code_or(orientation_layup)
        material_str = Layup.trans_code_ma(material_layup)
        layup_list = Layup.code_to_list(layup_str)
        material_list = Layup.code_to_list(material_str, flag='material')
        return cls(layup_str, material_str, layup_list, material_list)


class Material:
    def __init__(self, name, e1, e2, g12, v12, t, q=None):
        """

        Parameters
        ----------
        name : str
            Name of the defined material
        e1 : float
            Elastic modulus of ply, longitudinal direction
        e2 : float
            Elastic modulus of ply, transverse direction
        g12 : float
            Shear modulus of ply
        v12 : float
            Poisson's ratio of ply for loading in the 1 direction
        t : float
            Material thickness
        q : np.ndarray
            Q matrix of the material

        """
        self.name = name
        self.e1 = e1
        self.e2 = e2
        self.g12 = g12
        self.v12 = v12
        v21 = (v12/e1)*e2
        self.v21 = v21
        self.t = t
        if q is None:
            self.q = Material.return_q(e1, e2, g12, v12, v21)
        else:
            self.q = q

    def is_uni(self):
        uni_con = True
        if self.e1/self.e2 != 1.0:
            uni_con = False
        return uni_con

    @staticmethod
    def return_q(e1, e2, g12, v12, v21):
        q11 = e1/(1-(v12*v21))
        q12 = (v21*e1)/(1-(v12*v21))
        q22 = e2/(1-(v12*v21))
        q66 = g12
        q = np.array([[q11, q12, 0], [q12, q22, 0], [0, 0, q66]])
        return q

    @classmethod
    def gen_from_dict(cls, material_dict, name):
        # name = material_dict['name']
        e1 = material_dict['e1']
        e2 = material_dict['e2']
        g12 = material_dict['g12']
        v12 = material_dict['v12']
        t = material_dict['t']
        return cls(name, e1, e2, g12, v12, t)


class MatLib:
    def __init__(self, *args):
        """A class for forming a library of material definitions.

        Parameters
        ----------
        args : Material objects or dict
            A sequence of material objects or dictionary of material definitions

        Returns
        -------
        MatLib object
            An instance of the MatLib class

        """
        for material_obj in args:
            if isinstance(material_obj, Material):
                setattr(self, material_obj.name, material_obj)
            elif isinstance(material_obj, dict):
                for name, mat_dict in material_obj.items():
                    mat_obj = Material.gen_from_dict(mat_dict, name)
                    setattr(self, name, mat_obj)


class Laminate:
    """A class for defining laminate objects.

    """
    def __init__(self, layup_obj, mat_lib_obj, abd=None, effective_props=None, reqs=None):
        """

        Parameters
        ----------
        layup_obj
        mat_lib_obj
        abd
        effective_props
        reqs

        """
        self.layup = layup_obj
        self.mat_lib = mat_lib_obj
        if abd is None:
            self.abd = self.return_abd()
        else:
            self.abd = abd

        if effective_props is None:
            self.return_effective_props(reqs)
        else:
            self.effective_props = effective_props

    @staticmethod
    def balanced_solver(mat_lib_obj, req_obj):
        abd_matrix = 0
        effective_props = 0
        return abd_matrix, effective_props

    @staticmethod
    def return_t_sigma(theta):
        m = np.cos(theta*0.0174533)
        n = np.sin(theta*0.0174533)
        ms = m**2
        ns = n**2
        mn = m*n
        msmns = ms - ns
        t_sigma = np.array([[ms, ns, 2*mn], [ns, ms, -2*mn], [-mn, mn, msmns]])
        return t_sigma

    @staticmethod
    def return_t_epsilon(theta):
        m = np.cos(theta*0.0174533)
        n = np.sin(theta*0.0174533)
        ms = m ** 2
        ns = n ** 2
        mn = m * n
        msmns = ms - ns
        t_epsilon = np.array([[ms, ns, mn], [ns, ms, -mn], [-2*mn, 2*mn, msmns]])
        return t_epsilon

    @staticmethod
    def return_q_bar(material_obj, theta):
        q = material_obj.q
        t_sigma = Laminate.return_t_sigma(theta)
        t_epsilon = Laminate.return_t_epsilon(theta)
        t_sigma_inv = np.linalg.inv(t_sigma)
        q_bar = t_sigma_inv @ q @ t_epsilon
        return q_bar

    def return_total_t(self):
        layup_obj = self.layup
        mat_lib_obj = self.mat_lib
        t_list = []
        for material in layup_obj.material_list:
            mat_obj = getattr(mat_lib_obj, material)
            t_list.append(mat_obj.t)
        return t_list

    def return_thickness(self):
        t = sum(self.return_total_t())
        return t

    def gen_z_list(self):
        """

        Parameters
        ----------
        layup_obj : Layup object
            Layup object describing the laminate layup
        mat_lib_obj : MatLib object
            MatLib object describing the laminate layup

        Returns
        -------
        z_bar_list : list
            List of the z_bar values for each ply

        """
        t_list = self.return_total_t()
        t = sum(t_list)
        t2 = t/2
        z_list = []
        z_bar_list = []
        z_list.append(-t2)
        z_bar_list.append(-t2 + t_list[0] / 2.0)
        # Even number of plies
        if len(t_list) % 2 == 0:
            mid_ind = int(len(t_list)/2)
            for i in range(1, mid_ind):
                z = z_list[i-1] + t_list[i-1]
                z_bar = z + t_list[i]/2.0
                z_list.append(z)
                z_bar_list.append(z_bar)
            z_list.append(t_list[mid_ind])
            z_bar_list.append(t_list[mid_ind]/2.0)
            for i in range(mid_ind+1, len(t_list)-1):
                z = z_list[i-1] + t_list[i]
                z_bar = z - t_list[i] / 2.0
                z_list.append(z)
                z_bar_list.append(z_bar)
        # Odd number of plies
        else:
            mid_ind = int(np.floor(len(t_list)/2))
            print(mid_ind)
            for i in range(1, mid_ind):
                z = z_list[i-1] + t_list[i-1]
                z_bar = z + t_list[i]/2.0
                z_list.append(z)
                z_bar_list.append(z_bar)
            z_list.append(t_list[mid_ind]/2.0)
            z_bar_list.append(0.0)
            for i in range(mid_ind+1, len(t_list)-1):
                z = z_list[i-1] + t_list[i]
                z_bar = z - t_list[i]/2.0
                z_list.append(z)
                z_bar_list.append(z_bar)

        z_list.append(t2)
        z_bar_list.append(z_list[-1] - t_list[-1] / 2.0)
        return z_bar_list

    def gen_abd(self):
        """Generate the ABD matrix

        Parameters
        ----------
        layup_obj : Layup object
            A layup object specifying a laminate layup.
        mat_lib_obj : MatLib object
            A material library with definitions for each material in layup_obj

        Returns
        -------
        a_mat : np.array
            The A matrix for a laminate.
        b_mat : np.array
            The B matrix for a laminate.
        d_mat : np.array
            The D matrix for a laminate.

        """
        layup_obj = self.layup
        mat_lib_obj = self.mat_lib
        t_list = self.return_total_t()
        z_bar_list = self.gen_z_list()
        a_mat = np.zeros((3, 3))
        b_mat = np.zeros((3, 3))
        d_mat = np.zeros((3, 3))
        for i, (theta, material) in enumerate(zip(layup_obj.layup_list, layup_obj.material_list)):
            # print(theta, material)
            mat_obj = getattr(mat_lib_obj, material)
            q_bar = Laminate.return_q_bar(mat_obj, theta)
            a_mat = a_mat + q_bar*t_list[i]
            b_mat = b_mat + q_bar*t_list[i]*z_bar_list[i]
            d_mat = d_mat + q_bar*((t_list[i]**3.0)/12.0 + t_list[i]*z_bar_list[i]**2.0)
            # print(mat_obj.name)
        return a_mat, b_mat, d_mat

    def return_abd(self):
        a_mat, b_mat, d_mat = self.gen_abd()
        ab_mat = np.concatenate((a_mat, b_mat), axis=1)
        bd_mat = np.concatenate((b_mat, d_mat), axis=1)
        abd_mat = np.concatenate((ab_mat, bd_mat), axis=0)
        return abd_mat

    def is_lam_symmetric(self):
        """Test to determine if a laminate is symmetric.

        Returns
        -------
        b_test : float
            The ratio of the Lp,q norms of the B matrix w/r/t the Lp,q norm of the ABD matrix
        b_inv_test : float
            The ratio of the Lp,q norms of the b matrix w/r/t the Lp,q norm of the abd matrix

        """
        # Return the Lp,q norm of the ABD matrix for testing
        lpq_norm_abd = np.sqrt(np.sum(np.square(self.abd)))
        # Return the abd matrix (not to be confused with ABD)
        abd_mat_inv = np.linalg.inv(self.abd)
        # Return the Lp,q norm of the abd matrix for testing type
        lpq_norm_abd_inv = np.sqrt(np.sum(np.square(abd_mat_inv)))
        # Get the B and b matrices:
        b_mat = self.abd[0:3, 3:6]
        b_mat_inv = abd_mat_inv[0:3, 3:6]
        # Get the Lp,q norms of the B and b matrices:
        lpq_norm_b = np.sqrt(np.sum(np.square(b_mat)))
        lpq_norm_b_inv = np.sqrt(np.sum(np.square(b_mat_inv)))
        # Test Lp,q norms of B and b w/r/t Lp,q norm of ABD
        b_test = lpq_norm_b/lpq_norm_abd
        b_inv_test = lpq_norm_b/lpq_norm_abd_inv

        return b_test, b_inv_test

    def is_lam_balanced(self):
        # Get the A matrix
        a_mat = self.abd[0:3, 0:3]
        # Return the Lp,q norm of the A matrix for testing
        lpq_norm_a = np.sqrt(np.sum(np.square(a_mat)))
        # Return A16, A26 from the ABD matrix
        a16 = self.abd[0, 2]
        a26 = self.abd[1, 2]
        # Test a16 and a26 against Lp,q norm of A
        a16_test = a16/lpq_norm_a
        a26_test = a26/lpq_norm_a

        return a16_test, a26_test

    def is_lam_flex_balanced(self):
        # Get the D matrix:
        d_mat = self.abd[3:6, 3:6]
        # Return the Lp,q norm of the D matrix for testing
        lpq_norm_d = np.sqrt(np.sum(np.square(d_mat)))
        # Return A16, A26 from the ABD matrix
        d16 = self.abd[3, 5]
        d26 = self.abd[4, 5]
        # Test d16 and d26 against Lp,q norm of D
        d16_test = d16 / lpq_norm_d
        d26_test = d26 / lpq_norm_d

        return d16_test, d26_test

    def validate(self, req_obj=None):
        """

        Parameters
        ----------
        req_obj : Requirements
            Object representing requirements for the laminate.

        Returns
        -------
        sym_con : bool
            Is the laminate symmetric
        bal_con : bool
            Is the laminate balanced
        flex_con : bool
            Is the laminate flexurally balanced

        """
        # Lp,q norms
        b_t, b_i_t = self.is_lam_symmetric()
        a16_t, a26_t = self.is_lam_balanced()
        d16_t, d26_t = self.is_lam_flex_balanced()

        # is it symmetric
        sym_con = False
        if req_obj.symmetry is not None:
            if abs(b_t) < req_obj.symmetry and abs(b_i_t) < req_obj.symmetry:
                sym_con = True

        # is it balanced
        bal_con = False
        if req_obj.balanced is not None:
            if abs(a16_t) < req_obj.balanced and abs(a26_t) < req_obj.balanced:
                bal_con = True

        # is it flexurally balanced
        flex_con = False
        if req_obj.flex_bal is not None:
            if abs(d16_t) < req_obj.flex_bal and abs(d26_t) < req_obj.flex_bal:
                flex_con = True

        return {'symmetric': sym_con, 'balanced': bal_con, 'flexurally balanced': flex_con}

    def return_effective_props(self, req_obj=None):
        """

        Parameters
        ----------
        abd_mat : np.ndarray
            ABD Matrix of the laminate
        req_obj : Requirements
            Requirements threshold for determining various properties of the laminate

        Returns
        -------
        effective_props : dict
            Dictionary containing the effective properties of the laminate.

        """
        if req_obj is None:
            req_obj = Requirements(1e-6, 1e-6, 1e-6)
        t = self.return_thickness()
        sym_con, bal_con, flex_con = self.validate(req_obj)
        abd_mat_inv = np.linalg.inv(self.abd)
        a_inv = abd_mat_inv[0:3, 0:3]
        # unrestrained properties
        e_x = 1.0/(t*a_inv[0, 0])
        e_y = 1.0/(t*a_inv[1, 1])
        v_xy = -a_inv[0, 1]/a_inv[0, 0]
        v_yx = -a_inv[0, 1]/a_inv[1, 1]
        g_xy = 1.0/(t*a_inv[2, 2])
        unrestrained_props = {'e_x': e_x, 'e_y': e_y, 'v_xy': v_xy, 'v_yx': v_yx, 'g_xy': g_xy}
        effective_props = {'unrestrained': unrestrained_props}
        # restrained and neutral axis
        if sym_con is False:
            a_star = np.linalg.inv(self.abd[0:3, 0:3])
            ex_rc = 1.0/(t*a_star[0, 0])
            ey_rc = 1.0/(t*a_star[1, 1])
            vxy_rc = -a_star[0, 1]/a_star[0, 0]
            vyx_rc = -a_star[0, 1]/a_star[1, 1]
            gxy_rc = 1.0/(t*a_star[2, 2])
            rest_curv_props = {'e_x': ex_rc, 'e_y': ey_rc, 'v_xy': vxy_rc, 'v_yx': vyx_rc, 'g_xy': gxy_rc}
            effective_props['restrained_curvature'] = rest_curv_props
            d_inv = abd_mat_inv[3:6, 3:6]
            b_inv = abd_mat_inv[0:3, 3:6]
            ex_na = d_inv[0, 0]/(t*(a_inv[0, 0]*d_inv[0, 0] - b_inv[0, 0]**2.0))
            ey_na = d_inv[1, 1]/(t*(a_inv[1, 1]*d_inv[1, 1] - b_inv[1, 1]**2.0))
            gxy_na = d_inv[2, 2]/(t*(a_inv[2, 2]*d_inv[2, 2] - b_inv[2, 2]**2.0))
            neu_axis_props = {'e_x': ex_na, 'e_y': ey_na, 'g_xy': gxy_na}
            effective_props['neutral_axis'] = neu_axis_props
        self.effective_props = effective_props

    @classmethod
    def gen_balanced(cls, mat_lib_obj, req_obj):
        layup_obj = 0
        mat_lib_obj = 0
        abd_matrix = 0
        effective_props = 0
        return cls(layup_obj, mat_lib_obj, abd_matrix, effective_props)

    @classmethod
    def gen_bal_sym(cls, mat_lib_obj, req_obj):
        layup_obj = 0
        abd_matrix = 0
        effective_props = 0
        return cls(layup_obj, mat_lib_obj, abd_matrix, effective_props)

    @classmethod
    def gen_from_layup_mat_lib(cls, layup_obj, mat_lib_obj):

        return cls()


class Requirements:
    """Class for storing laminate requirements.

    Can be used for automated design or validation of user-defined laminates.

    """
    def __init__(self, symmetry=None, balanced=None, flex_bal=None):
        """

        Parameters
        ----------
        symmetry : float
            Tolerance on the Lqr norms for symmetry assumption
        balanced : float
            Tolerance on the Lqr norms for the balanced requirement
        flex_bal : float
            Tolerance on the Lqr norms for the flexurally balanced requirement
        """
        self.symmetry = symmetry
        self.balanced = balanced
        self.flex_bal = flex_bal



if __name__ == "__main__":
    # layup_orientations = "[(+-45/90:2/0):2/0\]:s"
    # layup_materials = "[(+-material1/material2:2/material2):2/material1\]:s"
    # layup_orientations = "[45/-45/0\]:s"
    # layup_materials = "[material1/material1/material1\]:s"
    # layup_orientations = "[0/(45/-45):2s/-45:2/90]:2s"
    # layup_materials = "[material1/(material1/material1):2s/-material1:2/material1]:2s"
    layup_orientations = "[0/45/-45/90]:s"
    layup_materials = "[material1/material1/material1/material1]:s"
    layup_1 = Layup.gen_from_short(layup_orientations, layup_materials)
    # print(test.layup_str)
    # print(test.material_str)
    # print(test.layup_list)
    # print(test.material_list)

    # test = Layup.trans_code_or(layup_orientations)
    # print(test)

    # test2 = Layup._fix_sy_np(layup_orientations)
    # print(test2)
    #
    # print(test.layup_str)
    # print(test.material_str)
    # print(test.layup_list)
    # print(test.material_list)
    # print(len(test.layup_list))
    # print(len(test.material_list))
    # test2 = Layup('[45/-45/0/-45/45]', '[fabric/-fabric/tape/-fabric/fabric]')
    #
    # print(test2.layup_str)
    # print(test2.material_str)
    # print(test2.layup_list)
    # # print(test2.material_list)
    #
    material1 = Material('material1', 70, 70, 35, 0.13, 0.1)
    material2 = Material('material2', 70, 72, 35, 0.13, 0.15)
    material3 = Material('material3', 73, 73, 35, 0.13, 0.2)

    print(material1.is_uni())
    print(material2.is_uni())
    print(material2.e1, material2.e2)
    print(material3.is_uni())
    # print(type(material1))
    #
    library1 = MatLib(material1, material2, material3)

    laminate1 = Laminate(layup_1, library1)

    print(np.array2string(laminate1.abd, max_line_width=np.inf))
    print(np.array2string(laminate1.abd[3:6, 3:6], max_line_width=np.inf))

    print(laminate1.effective_props)
    req_obj = Requirements(1e-6, 1e-6, 1e-6)
    print(laminate1.validate(req_obj))

    # # print(test3.abd)
    # print(np.array2string(test3.abd, max_line_width=np.inf))
    # print(np.array2string(test3.abd, max_line_width=np.inf))
    #
    # print(test3)
    # # print(test3.is_lam_balanced())
    # # print(test3.is_lam_symmetric())
    # print(test3.is_lam_flex_balanced())
    #
    # reqs = Requirements(1e-6, 1e-6, 1e-6)
    #
    # print(test3.validate(reqs))
    #
    # print(test3.effective_props)

    # print(test3.abd)

    #
    # # for attr, value in library1.__dict__.items():
    # #     print(attr, 'properties:')
    # #     for attrm, valuem in value.__dict__.items():
    # #         print(attrm, '=', valuem)
    #
    # # for i, (angle, mat) in enumerate(zip(test.layup_list, test.material_list)):
    # #     print(angle, mat, i)
    # #     props = []
    # #     mat_obj = getattr(library1, mat)
    # #     print(mat_obj.name)
    # #     # print(props)
    #     # for attr, value in getattr(library1, material).__dict__.items():
    #     #     props.append([attr, value])
    #     # print(props)
    #     # print(getattr(library1, material))
    #
    # t_list = Laminate.return_total_t(test, library1)
    #
    # z_list_bar = Laminate.gen_z_list(test, library1)
    # print(t_list)
    # print(len(t_list))
    # # print(z_list)
    # # print(len(z_list))
    # print(z_list_bar)
    # # print(t)
    # # print(type(t))
    # #
    # # print(t_list)
    # #
    # # print(z_list)
    # #
    # # print(len(t_list))
    # #
    # # print(len(z_list))
    #
    # a, b, d = Laminate.gen_abd(test, library1)
    #
    # test_laminate = Laminate(test, library1)
    #
    # print('ABD Matrix:')
    # print(np.array2string(test_laminate.abd, max_line_width=np.inf))
    # print('A16 = ', test_laminate.abd[0, 2])
    # print('A26 = ', test_laminate.abd[1, 2])
    # print('B Matrix:')
    # B = test_laminate.abd[0:3, 3:6]
    # print(np.array2string(B, max_line_width=np.inf))
    #
    # B_test, b_test = Laminate.is_lam_symmetric(test_laminate.abd)
    # print("B test was:", B_test)
    # print("b test was:", b_test)
    #
    # a16_test, a26_test = Laminate.is_lam_balanced(test_laminate.abd)
    # print("A16 test was:", a16_test)
    # print("A26 test was:", a26_test)
    #
    # # print('ABD Matrix:')
    # # print(np.array2string(test_laminate.abd, max_line_width=np.inf))
    # # # print(np.array2string(test_laminate.abd[0:3, 0:3], max_line_width=np.inf))
    # # print('D Matrix:')
    # # print(np.array2string(test_laminate.abd[3:6, 3:6], max_line_width=np.inf))
    # #
    #
    # # print(material1.q)
    # # norm = Laminate.is_lam_symmetric(test_laminate.abd)
    # # abd_inv = np.linalg.inv(test_laminate.abd)
    # # norm2 = Laminate.is_lam_symmetric(abd_inv)
    # # print(norm)
    # # print(norm2)
    # # abd_inv = Laminate.is_lam_symmetric(test_laminate.abd)
    # #
    # # print('abd Matrix:')
    # # print(np.array2string(abd_inv, max_line_width=np.inf))
    # # print('B Matrix:')
    # # B_sm = abd_inv[0:3, 3:6]
    # # print(np.array2string(B_sm, max_line_width=np.inf))
    #
    #
    # # for row in test_laminate.abd:
    # #     print(row)
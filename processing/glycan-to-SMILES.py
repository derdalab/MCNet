#!/usr/bin/python3

import sys
import collections
import re

spacers = {'PA':   'Nc1ccccn1',
           'lipid':'O[Li]',
           'PGL':  'O-c(cc1)ccc1[Li]', # phenolic glcolipid  ACS Chem Bio 2017, 12 2990-3002
           'Sp0':  'OCCN-C(=O)CCCOC',
           'Sp8':  'OCCCN-C(=O)CCCOC',
           'Sp9':  'OCCCCCN-C(=O)CCCOC',
           'Sp10': 'NC(=O)CN-C(=O)CCCOC',
           'Sp11': 'OCc1ccc(cc1)NC(=O)CN-C(=O)CCCOC',
           'Sp12': 'NC(=O)C[C@@H](C(=O)O)N-C(=O)CCCOC',
           'Sp13': 'OC(=O)CN-C(=O)CCCOC',
           'Sp14': 'O[C@H](C)[C@@H](C(=O)O)N-C(=O)CCCOC',
           'Sp15': 'OC[C@@H](C(=O)O)N-C(=O)CCCOC',
           'Sp16': 'Oc1ccc(cc1)N-C(=O)CCCOC',
           'Sp17': 'OCc1ccc(cc1)N-C(=O)CCCOC',
           'Sp18': 'OCCCNC(=O)CCCCCN-C(=O)CCCOC',
           'Sp19': 'NC(=O)C[C@@H](C(=O)O)N-C(=O)[C@H](CCC(=O)O)N-C(=O)CCCOC', # -Glu-Asn
           'Sp19a':'NC(=O)C[C@@H](C(=O)-N[C@@H](CCCCN)C(=O)O)N-C(=O)CCCOC', # Asn-Lys
           'Sp20': 'NC(=O)C[C@@H](C(=O)-N[C@@H](CCCN=C(N)N)C(=O)O)N-C(=O)[C@H](CCC(=O)O)N-C(=O)CN-C(=O)CCCOC',
           'Sp21': 'N(C)OCCN-C(=O)CCCOC',
           'Sp22': 'NC(=O)C[C@@H](C(=O)-N[C@@H](CO)C(=O)-N[C@@H]([C@H](O)C)C(=O)O)N-C(=O)CCCOC',
           'Sp23': 'OCCOCCOCCOCCOCCOCCN-C(=O)CCCOC',
           'Sp24': 'NC(=O)C[C@@H](C(=O)-N[C@@H](CCCCN)C(=O)-N[C@@H]([C@H](O)C)C(=O)O)N-C(=O)[C@H](C)N-C(=O)[C@H](C(C)C)N-C(=O)[C@H](CCCCN)N-C(=O)CCCOC',
           'Sp25': 'NC(=O)C[C@@H](C(=O)-N[C@@H](CCCCN)C(=O)O)N-C(=O)[C@H](C)N-C(=O)[C@H](C(C)C)N-C(=O)CCCOC',
           'MDPLys':'O[C@H]([C@H](O1)CO)[C@@H]([C@@H](NC(=O)C)[C@@H]1O)O[C@H](C)C(=O)-N[C@@H](C)C(=O)-' +
                    'N[C@@H](C(=O)N)CCC(=O)-N[C@@H](C(=O)O)CCCCN-C(=O)CCCOC',  # muramic dipeptide (MurNAc + Ala + iThr + Lys)
           'O':'O',
           '':''}



lookup = {'All': (6, 1, 5, 'D', [True, True, True]),	# beta-D-allose
          'Alt': (6, 1, 5, 'L', [False, True, True],),	# beta-L-altrose
          'Man': (6, 1, 5, 'D', [False, False, True]),	# beta-D-mannose
          'Glc': (6, 1, 5, 'D', [True, False, True]),	# beta-D-glucose
          'Gal': (6, 1, 5, 'D', [True, False, False]),  # beta-D-galactose
          'Tal': (6, 1, 5, 'D', [False, False, False]),	# beta-D-talose
          'Ido': (6, 1, 5, 'L', [False, True, False]),	# beta-L-idose
          'Gul': (6, 1, 5, 'D', [True, True, False]),	# beta-D-gulose
          'Rib': (5, 1, 5, 'D', [True, True]),		# beta-D-ribose
          'Ara': (5, 1, 5, 'L', [False, True]),		# beta-L-arabinose
          'Lyx': (5, 1, 5, 'D', [False, False]),	# beta-D-lyxose
          'Xyl': (5, 1, 5, 'D', [True, False]),		# beta-D-xylose
          'ManHep': (7, 1, 5, 'DD', [False, False, True, True, True]),
          'manHep': (7, 1, 5, 'DD', [False, False, True, True, True]),
          'Dha': (7, 2, 6, 'D', [None, None, False, False, True]),
          'Kdo': (8, 2, 6, 'DD', [None, None, False, False, True, True]),
          'Kdn': (9, 2, 6, 'DD', [None, None, True, False, False, True, True])
          }

def sugar(name = "?", anomer = 'beta', chiral = None):
    #print('Debug:', name, anomer, chiral)
    if name in lookup:
        s = mono(lookup[name][0])
        s.ring_carbon = (lookup[name][1],)
        s.ring_oxygen = (lookup[name][2],)
        s.chiral[-2] = True
        for i, c in enumerate(lookup[name][4], start = 1):
            s.chiral[i] = c
        if anomer == 'beta' or anomer == 'β':
            s.chiral[s.ring_carbon[0] - 1] = True
        elif anomer == 'alpha' or anomer == 'α':
            s.chiral[s.ring_carbon[0] - 1] = False
        if name in ['Dha', 'Kdo', 'Kdn']:
            s.sidechain[0] = '(=O)O'
            s.hydrogens[1] = 0
            s.deoxy(pos = 3)
        if name in ['Dha',]:
            s.sidechain[7-1] = '(=O)O'
            s.hydrogens[7-1] = 0
        #print('CCC:', chiral, name)
        # TODO handle double chiral indicators in general
        if chiral == 'L' or (chiral == None and lookup[name][3] == 'L'):
            s.chiral = [not x if x != None else None  for x in s.chiral]
        if chiral is not None and len(chiral) > 1:
            if chiral[0] == 'L' and name in ['manHep', 'ManHep']:
                s.chiral[5:] = [not x if x != None else None  for x in s.chiral[5:]]
            if chiral[1] == 'L' and name in ['manHep', 'ManHep']:
                s.chiral[:5] =[not x if x != None else None  for x in s.chiral[:5]]
            
        return s
    else:
        #print('Logging', name, chiral, anomer)
        if name == 'ha' and chiral[-1] == 'D':
            return sugar('Dha', anomer, chiral[:-1])
        elif name == 'ig' and chiral[-1] == 'D':
            return sugar('Dig', anomer, chiral[:-1])
        elif name == 'eg' and chiral[-1] == 'L':
            return sugar('Leg', anomer, chiral[:-1])
        elif name == 'yx' and chiral[-1] == 'L':
            return sugar('Lyx', anomer, chiral[:-1])


        
        if name[-3:] == 'OAc':
            s = sugar(name[:-3], anomer, chiral)
            s.OAc()
            return s
        elif name[-4:] == 'fOAc':
            s = sugar(name[:-4], anomer, chiral)    # TODO furanose ring
            s.OAc()
            return s
        elif name[-4:] == 'OAcA':
            s = sugar(name[:-4], anomer, chiral)
            s.OAc()
            s.A()
            return s
        elif name[-3:] == 'NAc':
            s = sugar(name[:-3], anomer, chiral)
            s.NAc()
            return s
        elif name[-4:] == 'NAcA':
            s = sugar(name[:-4], anomer, chiral)
            s.NAc()
            s.A()
            return s
        elif name[-1:] == 'A':
            s = sugar(name[:-1], anomer, chiral)
            s.A()
            return s
        elif name[-1:] == 'N':
            s = sugar(name[:-1], anomer, chiral)
            s.N()
            return s
        elif name[-2:] == 'AN' or name[-2:] == 'NA':
            s = sugar(name[:-2], anomer, chiral)
            s.N()
            s.A()
            return s
        elif name[-5:] == 'N(Gc)':
            s = sugar(name[:-5], anomer, chiral)
            s.sidechain[1] = 'NC(=O)CO'
            return s
        elif name[-3:] == 'NGc':
            s = sugar(name[:-3], anomer, chiral)
            s.sidechain[1] = 'NC(=O)CO'
            return s
        elif name[-4:] == 'NGcA':
            s = sugar(name[:-4], anomer, chiral)
            s.sidechain[1] = 'NC(=O)CO'
            s.a()
            return s
        elif name[-1:] == 'f':
            s = sugar(name[:-1], anomer, chiral)
            s.ring_carbon = (1,)
            s.ring_oxygen = (4,)
            return s 

        elif name == 'aGal':
            s = sugar('Gal', anomer = anomer, chiral = chiral)
            s.sidechain[3-1] = ''     
            s.ring_carbon = (3,)
            s.ring_oxygen = (6,)
            return s
        elif name == 'aMan':
            s = sugar('Man', anomer = anomer, chiral = chiral)
            s.sidechain[2-1] = ''     
            s.ring_carbon = (2,)
            s.ring_oxygen = (5,)
            return s
        elif name == 'Abe':
            s = sugar('Gal', anomer = anomer, chiral = chiral)
            s.deoxy()
            s.deoxy(3)
            return s
        elif name == 'Bac':
            s = sugar("Glc", anomer = anomer, chiral = chiral)
            s.N()
            s.N(pos = 4)
            s.deoxy()
            return s
        elif name == 'Col':
            s = sugar('Gal', anomer = anomer, chiral = 'L' if chiral == None else chiral)
            s.deoxy()
            s.deoxy(3)
            return s
        elif name == 'Dig':
            s = sugar('All', anomer = anomer, chiral = chiral)
            s.deoxy()
            s.deoxy(2)
            return s
        elif name == 'Fru':
            s = sugar('Glc', anomer = anomer, chiral = chiral) 
            s.chiral[:2] = [None, None]    
            s.ring_carbon = (2,)
            s.ring_oxygen = (6,)
            return s
        elif name == 'Tag':
            s = sugar('Gal', anomer = anomer, chiral = chiral) 
            s.chiral[:2] = [None, None]    
            s.ring_carbon = (2,)
            s.ring_oxygen = (6,)
            return s
        elif name == 'Sor':
            s = sugar('Gul', anomer = anomer, chiral = 'L' if chiral == None else chiral)
            s.chiral[:2] = [None, None]    
            s.ring_carbon = (2,)
            s.ring_oxygen = (6,)
            return s
        elif name == 'Psi':
            s = sugar('All', anomer = anomer, chiral = chiral) 
            s.chiral[:2] = [None, None]    
            s.ring_carbon = (2,)
            s.ring_oxygen = (6,)
            return s
        elif name == 'Fuc':
            s = sugar("Gal", anomer = anomer, chiral = 'L' if chiral == None else chiral)
            s.deoxy()
            return s
        elif name == 'Mur':
            s = sugar('Glc', anomer = anomer, chiral = chiral)
            s.N()
            s.sidechain[3-1] = 'O[C@H](C)C(=O)O'
            return s
        elif name == 'Oli':
            s = sugar('Qui', anomer = anomer, chiral = chiral)
            s.deoxy(2)
            return s
        elif name == 'Par':
            s = sugar('Qui', anomer = anomer, chiral = chiral)
            s.deoxy(3)
            return s
        elif name == 'Rha':
            s = sugar("Man", anomer = anomer, chiral = 'L' if chiral == None else chiral)
            s.deoxy()
            return s
        elif name == 'Qui':
            s = sugar("Glc", anomer = anomer, chiral = chiral)
            s.deoxy()
            return s
        elif name == 'Tyv':
            s = sugar('Man', anomer = anomer, chiral = chiral)
            s.deoxy()
            s.deoxy(3)
            return s
        elif name == 'Aci':
            s = sugar('Leg', anomer = anomer, chiral = chiral)
            s.epimer(pos = 7)
            s.epimer(pos = 8)
            return s
        elif name == 'Leg':
            s = sugar('Neu', anomer = anomer, chiral = chiral)
            s.deoxy(pos = 9)
            s.N(pos = 7)
            return s
        elif name == 'Neu':
            s = sugar('Kdn', anomer = anomer, chiral = chiral)
            s.sidechain[0] = '(=O)O'
            s.deoxy(pos = 3)
            s.N(pos = 5)
            s.ring_carbon = (2,)
            s.ring_oxygen = (6,)
            return s
        elif name == 'Pse':
            s = sugar('Aci', anomer = anomer, chiral = chiral)
            s.epimer(pos = 5)
            return s
        elif name == 'Neu5Ac':
            s = sugar('Neu', anomer = anomer, chiral = chiral)
            s.sidechain[0] = '(=O)O'
            s.deoxy(pos = 3)
            s.NAc(pos = 5)
            #s.ring_carbon = (2,)
            #s.ring_oxygen = (6,)
            return s
        elif name == 'Neu5Gc':
            s = sugar('Neu5Ac', anomer = anomer, chiral = chiral)
            s.sidechain[5-1] = 'NC(=O)CO'
            return s
        elif name == 'Neu5,9Ac2':
            s = sugar('Neu5Ac', anomer = anomer, chiral = chiral)
            s.NAc(pos = 9)
            return s
        #elif name == 'MurNAc':
        #    s = sugar('Glc', anomer = anomer, chiral = chiral)
        #    s.sidechain[3-1] = 'O[C@H](C)C(=O)O'
        #    s.NAc()
        #   return s
        elif name == 'Api':     #TODO L-Api (furanose)
            s = mono(4)
            s.chiral = [None, True, False, None]
            s.ring_carbon = (1,)
            s.ring_oxygen = (4,)
            s.sidechain[3-1] = 'O)(CO'
            s.hydrogens[3-1] = 0
            return s
        elif name == 'HexA':
            s = mono(6)
            s.ring_carbon = (1,)
            s.ring_oxygen = (5,)
            s.A()
            return s
        elif name == 'ΔUA':
            s = sugar('HexA')   # TODO fix this - might be based on GlcA
            s.sidechain[4-1] = 'X' # no double bonds - but mark location
            return s
        elif name == 'G-ol' or name == 'Sorbitol':
            s = sugar('Glc')   # TODO fix this
            s.chiral[1-1] = None
            s.ring_carbon = ()  # open the ring
            s.ring_oxygen = ()
            return s
        elif name == 'MDPLys':
            s = sugar('MurNAc')
            s.sidechain[3-1] = ('O[C@H](C)C(=O)' + '-N[C@@H](C)C(=O)' +
                    '-N[C@@H](C(=O)N)CCC(=O)' + '-N[C@@H](C(=O)O)CCCCN ')
            return s
        else:
            return None

class mono:
    def __init__(self, carbons = 6):
        self.chiral = [None] * carbons
        self.sidechain = ['O'] * carbons
        self.hydrogens = [1] * carbons
        self.ring_carbon = (1,)
        self.ring_oxygen = (5,)
        ###self.sidechain = ['O{}'.format(i)  for i in range(1, carbons + 1)]

    def carbon_string(self, i, flip, loop = 1):	# return string for carbon no. i (1-based)
        ring = str(loop)  if i in self.ring_carbon  else  ''
        hydrogen = 'H' if self.hydrogens[i-1] > 0 else ''
        if self.chiral[i-1] == None:
            return 'C' + ring
        elif (self.chiral[i-1] and not flip) or (not self.chiral[i-1] and flip):
            return '[C@' + hydrogen + ']' + ring
        else:
            return '[C@@' + hydrogen + ']' + ring

    def sidechain_string(self, i, loop = 1, fwd = True): # sidechain i (1-based) as string
        ring = str(loop)  if i in self.ring_oxygen  else  ''
        string = self.sidechain[i-1]
        #if not fwd:  #TODO

        return string + ring

    def backbone_run(self, start, end, loop = 1, at_end = True):
        direction = (-1) ** (start > end)   # == +/-1
        pairs = [(self.carbon_string(i, start>end, loop), self.sidechain_string(i, loop))
                                     for i in range(start, end, direction)]
        if len(pairs) == 0:
            part_list = []
        else:
            part_list = [c + '(' + s + ')'  if s != '' else c  for c, s in pairs]
            if at_end:
                part_list[-1] = pairs[-1][0] + pairs[-1][1] # remove ()'s on end nodes
        return ''.join(part_list)

    def print(self, start = 1, end = None, loop = 1):
        #print('Log:', start, end, self.sidechain)
        if start == None or start == '?':
            start = 1 if 2 * end > len(self.chiral) else len(self.chiral)
            if start == 6 and self.sidechain[5] == '(=O)O':   # sidechains are not reversed
                start = 5
            elif start == 7 and self.sidechain[6] == '(=O)O':   # sidechains are not reversed
                start = 6
            elif start == 9 and self.sidechain[8] == 'NC(=O)C':
                start = 8
            elif start == 5 and self.sidechain[4] == 'SC':
                start = 4
        elif type(start) == str:
            start = int(start)
        if end == None:
            end = len(self.chiral) if 2 * start < len(self.chiral) else 1
        elif type(end) == str:
            end = int(end)
        #print('SSS:', self.sidechain[start-1])
        #print('Log:', start, end)
        a = self.sidechain_string(start, loop, fwd = False)
        b = self.carbon_string(start, bool(start < end) ^ bool(start == 1), loop)
        e = self.carbon_string(end,   bool(start < end) ^ bool(end == 1), loop)
        g = self.sidechain_string(end, loop)
        if start < end:
            c = self.backbone_run(start - 1, 0, loop)
            d = self.backbone_run(start + 1, end, loop, at_end = False)
            f = self.backbone_run(end + 1, len(self.chiral) + 1, loop)
        else:
            c = self.backbone_run(start + 1, len(self.chiral) + 1, loop)
            d = self.backbone_run(start - 1, end, loop, at_end = False)
            f = self.backbone_run(end - 1, 0, loop)
        
        if c != '':                 # c needs parentheses - but omit if empty
            c = '(' + c + ')'
        if f == '' or g == '':      # f needs parentheses - optional if f or g is empty
            fg = f + g
        else:
            fg = '(' + f + ')' + g

        return '{}{}{}{}{}{}'.format(a, b, c, d, e, fg)

    def N(self, pos = 2):
        self.sidechain[pos-1] = 'N'
    def OAc(self, pos = 2):
        self.sidechain[pos-1] = 'OC(=O)C'
    def NAc(self, pos = 2):
        self.sidechain[pos-1] = 'NC(=O)C'
    def A(self, pos = 6):
        self.sidechain[pos-1] = '(=O)O'
    def deoxy(self, pos = 6):
        self.sidechain[pos-1] = ''
        self.chiral[pos-1] = None
    def epimer(self, pos):
        self.chiral[pos-1] = not self.chiral[pos-1]




# monosaccharide_names = ['Fuc', 'GalNAc', 'Gal', 'GlcA', 'GlcNAc', 'GlcN[Gc]', 'Glc', 'G-ol', 
#                         'Man', 'Neu5,9Ac2', 'Neu5Ac', 'Neu5Gc', 'KDN']
# prefixes = ['(3S)', '4S(3S)', '(4S)', '(6S)', '6S(3S)', '(6S)(4S)', '(6P)']
# anomeric_descriptors = ['(a1-2)', '(a1-3)', '(a1-4)', '(a1-6)',
#                         '(a2-3)', '(a2-6)', '(a2-8)',
#                         '(b1-2)', '(b1-3)', '(b1-4)', '(b1-6)', '(b2-6)']

# if one entry is a prefix of another, the shorter (prefix) one must occur later in
# the list or instances of the longer one will be incompletely matched
prefix = re.compile(r'\(([346])([SP])\)')
#prefix = re.compile(r'([346][SP]|\([346][SP]\)?\([346][SP]\)([1-9](,[1-9])*[de])?')
#prefix = re.compile(r'([346])([SP])|\(([346])([SP])\)|([1-9](,[1-9])*[de])')
prefixDL = re.compile(r'(([DL]+)-?)?')
prefix = re.compile(r'([1-9](,[1-9])*)([de])')
suffix = re.compile(r'([1-9](,[1-9])*)(OS|OP|OMe|NAc|NS|Ac|OAc|SH|SMe|NBz|MeA|Me|acyc)([2-9]?)')
###monosaccharide_names = re.compile(r'aGal|aMan|Araf|Ara|Fuc|Fru|GalA|GalNAc|Galf|Gal|GlcA|GlcNAc|GlcNGc|GlcN\(Gc\)|GlcN|Glc|G-ol|IdoA|Kdn|KDN|Man|MurNAc|Neu5,9Ac2|Neu5Ac|Neu5Gc|Neu|Qui|Rha|Tal|Xylf|Xyl|ΔUA|HexA|Sorbitol|MDPLys')
monosaccharide_names = re.compile(r'(a?)(Abe|Aci|All|Alt|Api|Ara|Bac|Col|DDManHep|Dha|ha|Dig|ig|Fuc|Fru|Gal|Glc|Gul|Ido|Kdn|Kdo|Leg|eg|Lyx|yx|ManHep|manHep|Man|Mur|Neu5,9Ac2|Neu|Oli|Par|Pse|Psi|Qui|Rib|Rha|Sor|Tag|Tal|Tyv|Xyl|ΔUA|G-ol|Hex|Sorbitol|MDPLys|KDN)(fOAc|OAcA|OAc|NAcA|NAc|NA|NGcA|NGc|N\(Gc\)|N|5Gc|5Ac|AN|A)?(f?)')
#linkage = re.compile(r' *\(([abαβ?])([12])-(S-)?([23456789])\) *')
linkage = re.compile(r'\(([abαβ?])([12])[-–](S[-–])?([23456789])\)')
#anchor = re.compile(r'\((([abαβ?])([12]?))?[-–](Sp19a|Sp[0-9]*|S[68]|P4|MDPLys)?')
anchor = re.compile(r'\((([abαβ?])?([12]?))?[-–](Sp19a|Sp[0-9]*|MDPLys|PA|lipid|PGL)?')

class Node:
    def __init__(self, anomer, chirality, label, modification, children):
        self.label = label
        self.anomer = anomer
        self.chirality = chirality
        self.modification = modification
        self.children = children

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def new_key(self):       # generate a new unique key in the graph dictionary
        if len(self.nodes) == 0:
            new_key = 0
        else:
            new_key = 1 + max(self.nodes.keys())
        return new_key
 
    def add_node(self, anomer, chirality, label, modification, children):
        new_key = self.new_key()
        
        # initialize new node with empty child list for directed graph
        self.nodes[new_key] = Node(anomer, chirality, label, modification, [])

        # connect to the new node from all the parent nodes
        for node, linkage in children.items():
            self.nodes[new_key].children.append(node)
            self.edges[(node, new_key)] = linkage

        return new_key

    def print(self):
        for i, (k, x) in enumerate(self.nodes.items()):
            print('JJJ:', i, k, x.label, x.anomer,x.modification, x.children)
        print(self.edges)

    # return a subgraph as a text string
    # input node is the rootmost node in the subgraph
    def subgraph_to_string(self, node, pos1):  #, anomer, exit):
        child_edges = [(child, self.edges[(child, node)])
                            for child in self.nodes[node].children]
        # TODO - cannonicalize branch order by sorting by decreasing attachment carbon
        #child_edges = sorted(child_edges, reverse = True, key = lambda x: int(x[1][-1]))

        #prior_list = ['{}-{})'.format(self.subgraph_to_string(child, edge[0]), edge[1])
        prior_list = ['{}-{}'.format(self.subgraph_to_string(child, edge[0]), edge[1])
                                                     for child, edge in child_edges] 

        if len(child_edges) == 0:
            next_carbon = None
        else:
            next_carbon = child_edges[0][1][1]

        if len(prior_list) == 0:
            prior_str = ''
        elif len(prior_list) == 1:
            prior_str = prior_list[0]
        else:
            prior_str = prior_list[0] + '[' + ']['.join(prior_list[1:]) + ']'
        #return (prior_str + self.nodes[node].modification + self.nodes[node].label +
        #           '(' + self.nodes[node].anomer + str(pos1))
        return (prior_str + ''.join(['('+str(x[0])+x[1]+')' for x in self.nodes[node].modification]) + self.nodes[node].label +
                    self.nodes[node].anomer + str(pos1))
#                   '(' + self.nodes[node].anomer + str(pos1))


    # return a subgraph as a SMILES string
    # input node is the rootmost node in the subgraph
    def subgraph_to_SMILES(self, node, pos1, loop, fwd):
        #print('SSS:', node, self.nodes[node].label, pos1, loop, fwd, self.nodes[node].children)
        child_edges = [(child, self.edges[(child, node)])
                            for child in self.nodes[node].children]
        # TODO - sort main branch first
        #child_edges = sorted(child_edges, reverse = True, key = lambda x: int(x[1][-1]))
        
        s = sugar(self.nodes[node].label, anomer = self.nodes[node].anomer,
                  chiral = self.nodes[node].chirality)

        # TODO - process node modifications
        ###print('LLL:', self.nodes[node].label)
        ###print('MMM:', self.nodes[node].modification)
        for mod_position, modification in self.nodes[node].modification:
            if modification == 'S' or modification == 'OS':
                s.sidechain[mod_position - 1] = 'OS(=O)(=O)O'
            elif modification == 'NS':
                s.sidechain[mod_position - 1] = 'NS(=O)(=O)O'
            elif modification == 'OP':
                s.sidechain[mod_position - 1] = 'OP(=O)(O)O'
            elif modification == 'Ac':
                s.sidechain[mod_position - 1] = 'C(=O)C'
            elif modification == 'NAc':
                s.sidechain[mod_position - 1] = 'NC(=O)C'
            elif modification == 'OAc':
                s.sidechain[mod_position - 1] = 'OC(=O)C'
            elif modification == 'SH':
                s.sidechain[mod_position - 1] = 'S'
            elif modification == 'Me' or modification == 'OMe':
                s.sidechain[mod_position - 1] = 'OC'
            elif modification == 'SMe':
                s.sidechain[mod_position - 1] = 'SC'
            elif modification == 'NBz':
                s.sidechain[mod_position - 1] = 'Nc1ccccc1'
            elif modification == 'MeA':
                s.sidechain[mod_position - 1] = 'CC(=O)O'
            elif modification == 'd':
                s.sidechain[mod_position - 1] = ''
                s.chiral[mod_position - 1] = None
            elif modification == 'e':
                s.chiral[mod_position - 1] = not s.chiral[mod_position - 1]
            elif modification == 'S':
                s.sidechain[mod_position - 1] = 'OS(=O)(=O)O'
            elif modification == 'P':
                s.sidechain[mod_position - 1] = 'OP(=O)(O)O'
            elif modification == 'acyc':
                s.chiral[s.ring_carbon-1] = None
                s.ring_carbon = ()  # open the ring
                s.ring_oxygen = ()
            else:
                print('Unknown modification:', modification)

        #while mods != '':
        #    mod_position = int(mods[1])
        #    if mods[2] == 'S':
        #        s.sidechain[mod_position - 1] = 'OS(=O)(=O)O'
        #        mods = mods[4:]
        #    elif mods[2] == 'P':
        #        s.sidechain[mod_position - 1] = 'OP(=O)(O)O'
        #        mods = mods[4:]

        for i, (child, edge) in enumerate(child_edges):
            next_loop = loop + 1  if i > 0 else  loop
            next_fwd = fwd and (i == 0)
            child_SMILES = self.subgraph_to_SMILES(child, edge[0], next_loop, next_fwd)
            if next_fwd:
                s.sidechain[edge[1]-1] = child_SMILES[:-1] + '-' + child_SMILES[-1] + '-'
            else:
                s.sidechain[edge[1]-1] = '-' + child_SMILES[0] + '-' + child_SMILES[1:]

        next_carbon = child_edges[0][1][1]  if len(child_edges) > 0  else  None
        ###print('MMM:', node, self.nodes[node].label, s.print(start = pos1, end = next_carbon))

        if fwd:
            return s.print(start = next_carbon, end = pos1, loop = loop)
        else: # reverse direction used for side chains
            return s.print(start = pos1, end = next_carbon, loop = loop)


    def to_string(self):  # special handling for the root node
        root = len(self.nodes) - 1
        child = self.nodes[root].children[0]
        return '{}-'.format(self.subgraph_to_string(child, self.edges[(child, root)][0]))

    def to_SMILES(self):  # special handling for the root node
        root = len(self.nodes) - 1
        child = self.nodes[root].children[0]
        if self.edges[(child, root)][0] != '?' and self.edges[(child, root)][0] != '':
            carbon1 = int(self.edges[(child, root)][0])
        elif self.nodes[child].label in ['KDN', 'Neu5Gc', 'Kdn', 'Neu5Ac', 'Neu5(Gc)', 'Neu5,9Ac2']:
            carbon1 = 2
        elif self.nodes[child].label in ['MDPLys']:
            carbon1 = 3
        else:
            carbon1 = 1
        return self.subgraph_to_SMILES(child, carbon1, 1, True)
   

def parse_glycan(the_graph, glycan_str, child_nodes, residue):
    if len(glycan_str) == 0:    # end of string, so return
       return (child_nodes, glycan_str, residue)
    elif glycan_str[0] == ']':  # end of side chain so return it
       return (child_nodes, glycan_str[1:], '')
    else:
        while glycan_str[0] == '[':     # begin sidechain
            side_chain, glycan_str, j = parse_glycan(the_graph, glycan_str[1:], {}, '')
            child_nodes.update(side_chain)   # add side chain to the connected children

        # read the prefix, sugar, and linkage for the next position
        chirality, mod_list, glycan_str = parse_prefix(glycan_str)
        monosaccharide, glycan_str      = parse_monosaccharide(glycan_str)
        #print('TTT', monosaccharide, '-', glycan_str)
        mod_list,       glycan_str = parse_suffix(glycan_str, mod_list)
        if len(glycan_str) == 0:
            anomer, c1, c2, link = '', '', '', 'O'  # linkage is optional at end of input
        else:
            anomer, c1, c2, glycan_str, link = parse_linkage(glycan_str)
        if link == 'S-':
            mod_list.append((c1, 'SH'))
            link = ''
        
        # add new node to graph
        ms_node = the_graph.add_node(anomer, chirality, monosaccharide, mod_list, child_nodes)

        # recursive call to handle the remainder of the string
        return parse_glycan(the_graph, glycan_str, {ms_node: (c1, c2)}, link)

# treat the prefix as a single entity
def parse_prefix(glycan_str):
    chirality = None
    mod_list = []
    prefix_result = prefixDL.match(glycan_str)
    if prefix_result.group(2) != None:
        #print('PPP:', prefix_result, prefix_result.groups(), prefix_result.group(1), prefix_result.group(2))
        chirality = prefix_result.group(2)
        glycan_str = glycan_str[prefix_result.span()[1]:]
    
    prefix_result = prefix.match(glycan_str)
    while prefix_result:
        #print('PPP:', prefix_result, prefix_result.groups())
        for position in prefix_result.group(1).split(','):
            mod_list.append((int(position), prefix_result.group(3)))
            #mod_list.append((int(position), prefix_result.group(2)))
        glycan_str = glycan_str[prefix_result.span()[1]:]
        prefix_result = prefix.match(glycan_str)
    #print('PPP:', mod_list)
    return chirality, mod_list, glycan_str

def parse_suffix(glycan_str, suffix_list):
    ####suffix_list = []
    suffix_result = suffix.match(glycan_str)
    #print('WWW:', glycan_str)
    while suffix_result:
        repeats = suffix_result.group(1).split(',')
        for position in repeats:
            suffix_list.append((int(position), suffix_result.group(3)))
        #print('WWW:', suffix_list)

        if len(repeats) == 1: # following digit not repeat count, but position of next mod
            matched_length = suffix_result.span()[1] - len(suffix_result.group(4))
        else:
            matched_length = suffix_result.span()[1]
        glycan_str = glycan_str[matched_length:]
        ###print('GGG:', glycan_str)
        suffix_result = suffix.match(glycan_str)
    return suffix_list, glycan_str

def parse_monosaccharide(glycan_str):
    ms_result = monosaccharide_names.match(glycan_str)
    if not ms_result:
        print('Monosaccharide not recognized: {} {}\n'.format(ms_result, glycan_str))
        return('', '')
    length = ms_result.span()[1]
    return (glycan_str[:length], glycan_str[length:])

def parse_linkage(glycan_str):
    linkage_result = linkage.match(glycan_str)
    #print('LLLP:', linkage_result, glycan_str)
    if linkage_result:
        (anomer, carbon1, link, carbon2) = linkage_result.groups()
        carbon1 = int(carbon1)
        carbon2 = int(carbon2)
    else:
        # look for the anchor if a standard linkage was not found
        ###link = None
        #print('LLL:', glycan_str)
        linkage_result = anchor.match(glycan_str)
        #print('AAAA:', linkage_result)
        if not linkage_result:
            print('Error: Linkage not recognized: "{}"\n'.format(glycan_str))
            return('', '')
        else:
            #print('JJJJ:', linkage_result.groups())
            (junk, anomer, carbon1, spacer) = linkage_result.groups()
            anomer  = anomer  if anomer  != None else '?'
            carbon1 = int(carbon1) if carbon1 != None and carbon1 != '' else '?'
            carbon2 = ''
            link = spacer
    # clean up the linkage string
    length = linkage_result.span()[1]
    anomer = anomer.replace("a", "α"). replace("b", "β")
    linkage_str = '{}{}-{}'.format(anomer, carbon1, carbon2)
    return (anomer, carbon1, carbon2, glycan_str[length:], link)

def glycan_string_to_graph(glycan_str, label):
    #print('GSG:', glycan_str, label)

    # represent the graph with a set of dictionaries
    the_graph = Graph()

    # parse the string into the graph structure
    anchor_node, remainder, residue = parse_glycan(the_graph, glycan_str, {}, '')
    ####print('AAA:', anchor_node) 
    ###print('AAA:', anchor_node)
    # special processing for the spacer node
    anchor_node = the_graph.add_node('', None, '$', '',
        {list(anchor_node.keys())[0]: list(anchor_node.values())[0]})

    if len(remainder) != 0:
         print("Error: incompletely parsed glycan:", glycan_str,
               '\n remainder =', len(remainder))
         return None
    #x = the_graph.to_string()
    ###print('XXX:', x)
    ###print('LLL:', x)
    #print(glycan_str)
    #s = the_graph.to_SMILES()
    #print('{}\t{}\t{}'.format(len(s), glycan_str, s))
    #while x[-1] in ['?', '-', '–']:
    #    x = x[:-1]
    ###print(x, glycan_str)
    if True: # or x == glycan_str[0:len(x)]:
        #print('RRR:', residue)

        if True or residue in spacers:
            #s = the_graph.to_SMILES()
            s = the_graph.to_SMILES()[:-1] + '-' + spacers[residue]
            #print('{}\t{}\t{}'.format(label, glycan_str, s))
            return s
#            print('{}\t{}\t{}\t{}'.format(len(s)+len(spacers[residue]), glycan_str, s, spacers[residue]))
        else:
            pass #print('Residue:', glycan_str[len(x):])
    else:
        pass #print('Mismatch:', x, glycan_str)
    ###the_graph.print()
    #print()



if False:
    glycan_string_to_graph('Xyl(-Sp0', 'TestA')
    glycan_string_to_graph('D-Xyl(-Sp0', 'TestB')
    glycan_string_to_graph('L-Xyl(-Sp0', 'TestC')
    glycan_string_to_graph('2,3dXyl(-Sp0', 'TestD')
    glycan_string_to_graph('L-2eXyl(b1-Sp0', 'TestE')

    glycan_string_to_graph('Xyl(-Sp0', 'TestF')
    glycan_string_to_graph('Xyl(b1-Sp0', 'TesG')
    glycan_string_to_graph('Xyl(-Sp0', 'TestH')

    glycan_string_to_graph('Glc(b-Sp0', 'TestI')
    glycan_string_to_graph('D-Glc(b-Sp0', 'TestJ')
    glycan_string_to_graph('L-Glc(b-Sp0', 'TestK')

    glycan_string_to_graph('Fuc(b-Sp0', 'TestL')
    glycan_string_to_graph('D-Fuc(b-Sp0', 'TestM')
    glycan_string_to_graph('L-Fuc(b-Sp0', 'TestN')


    glycan_string_to_graph('Araf(a1-5)Araf(-Sp0', 'Test1')
    glycan_string_to_graph('Glc4OS(-lipid', 'Test2')
    glycan_string_to_graph('Glc4Me(-lipid', 'Test3')
    glycan_string_to_graph('Xylf5SMe(-lipid', 'Test4')

    glycan_string_to_graph('Xylf5SMe(a1-4)Man(a1-5)Araf(b1-2)Araf(a1-3)[Araf(b1-2)Araf(a1-5)]Araf(a1-5)Araf(a1-lipid', 'Test5')
    glycan_string_to_graph('Xyl(-PGL', 'Test5')
    glycan_string_to_graph('Glc2,3Me2(-lipid', 'Test6')

    glycan_string_to_graph('Xyl(-lipid', 'Test7')
    glycan_string_to_graph('Ara(-lipid', 'Test8')


if sys.argv[1] != 'A':

    smiles = glycan_string_to_graph(sys.argv[1], sys.argv[1])
    print(sys.argv[1], smiles)
    sys.exit(0)


with open(sys.argv[2], 'r') as the_file:
    headers = the_file.readline()
    print('{}\t{}\t{}'.format('Name', 'IUPAC', 'SMILES'))
    for line in the_file.readlines():
        name, iupac = line.rstrip().split('\t')
        smiles = glycan_string_to_graph(iupac, name)
        print('{}\t{}\t{}'.format(name, iupac, smiles))

        


sys.exit(0)



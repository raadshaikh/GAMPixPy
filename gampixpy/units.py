# TODO: use a more sophisticated system for this.
# When do quntities acquere units?
# - in config file?
# - when config is loaded?
# - when additional parameters are computed?

"""

The basic units are :
    centimeter              (centimeter)
    second                  (second)
    Mega electron Volt      (MeV)
    positron charge         (e)
    Kelvin                  (kelvin)
    the amount of substance (mole)
    luminous intensity      (candela)
    radian                  (radian)
    steradian               (steradian)

"""

# distance
km = 1.e5
m = 1.e2
cm = 1.
mm = 1.e-1
um = 1.e-4
nm = 1.e-7

# time
h = 3600.
m = 60.
s = 1.
ms = 1.e-3
us = 1.e-6
ns = 1.e-9

# energy
GeV = 1.e3
MeV = 1.
keV = 1.e-3
eV = 1.e-6
meV = 1.e-9

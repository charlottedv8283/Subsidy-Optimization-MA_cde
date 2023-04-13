# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:07:15 2017
@author: jte-sre
"""
from __future__ import division

import numpy as np


def reference_building(building, grdtgszhl):
    """
    This function calculates parameters for a reference building with the
    dimensions of the examined building. Especially the values for the primary
    energy requirement and the transmission heat losses are needed to check if
    the examined building satisfies the requirement for subsidies in accordance
    with the KfW program for efficient buildings

    Parameters
    ----------
    building : dictionary
        Information about the dimensions of the building

    Returns
    -------
    reference_building : dictionary
        Informations about the primary energy requirement and the transmission
        heat losses of the reference building
    """

    # %% Define Parameters:

    component_size = {}

    component_size["Area"] = building["Area"]
    component_size["Volume"] = 0.76 * building["Area"] * building["Volume"]
    component_size["OuterWall"] = building["Area"] * building["OuterWall"]
    component_size["Rooftop"] = building["Area"] * building["Rooftop"]
    component_size["GroundFloor"] = building["Area"] * building["GroundFloor"]
    component_size["Window"] = building["Area"] * building["Window"]

    component_size["Window_south"] = 0.25 * component_size["Window"]
    component_size["Window_north"] = 0.25 * component_size["Window"]
    component_size["Window_east"] = 0.25 * component_size["Window"]
    component_size["Window_west"] = 0.25 * component_size["Window"]

    # Calculation for one heating period
    t = 185  # days

    # U-Values for EnEV reference building (W/m²K)
    # checked: GEG 2020 Anlage 1
    u_ref = {}
    u_ref["Window"] = 1.3
    u_ref["OuterWall"] = 0.28
    u_ref["GroundFloor"] = 0.35
    u_ref["Rooftop"] = 0.2

    Fx = {}
    Fx["Window"] = 1
    Fx["OuterWall"] = 1
    Fx["GroundFloor"] = 0.8
    Fx["Rooftop"] = 1

    # %% Starting Calculation

    # %% Losses

    # Transmisison heating losses
    building_parts = ("Window", "OuterWall", "Rooftop", "GroundFloor")
    H_t = sum(component_size[i] * u_ref[i] * Fx[i]
              for i in building_parts)  # W/K

    # Surcharge für thermal bridge
    # SOLVED: q.content
    # according to DIN V 4108-6 5.5.2.2. b) U_wb should be 0.1
    # --> checks in accordance with GEG Anlage 1 - 2
    U_wb = 0.05
    H_t = H_t + U_wb * sum(component_size[i] for i in building_parts)  # W/K

    # Specific Transmissison losses
    H_t_spec = H_t / sum(component_size[i] for i in building_parts)  # W/m²K

    # Ventilation losses
    # Still valid for GEG 2020
    # n could be 0.6 but 0.7 is assessment to safe side
    ro_cp = 0.34  # Wh/m³K
    n = 0.7  # 1/h
    H_v = ro_cp * n * component_size["Volume"]  # W/K

    # Total heating losses
    # G_t = 2900  # Kd - Gradtagzahl
    G_t = grdtgszhl
    f_na = 0.95  # Parameter for switching off the heater in the night
    f_ql = 0.024 * G_t * f_na
    Q_l = (H_t + H_v) * f_ql  # kWh

    # %%Profits

    # Annual solar radiation per direction
    I_s = 270  # kWh/m²a
    I_e = 155  # kWh/m²a
    I_w = 155  # kWh/m²a
    I_n = 100  # kWh/m²a

    # Solar profits
    F_f = 0.7  # reductoin becaus of window frame
    F_s = 0.9  # shading coefficient
    F_c = 1.0  # reduction because of sun protection
    g = 0.6 * 0.9  # reduction because of not-vertical radiation

    Q_s = F_f * F_s * F_c * g * (I_s * component_size["Window_south"] +
                                 I_w * component_size["Window_west"] +
                                 I_e * component_size["Window_east"] +
                                 I_n * component_size["Window_north"])
    # kWh/HP - Solar gains per heating period

    # Internal profits
    q_i = 5  # W/m²
    Q_i = 0.024 * q_i * component_size["Area"] * t  # kWh/a

    Q_g = Q_i + Q_s  # kWh/HP - Internal gains per heating period

    # %% Total heating demand

    eta = 0.95  # utilisation factor for heating gains
    Q_h = Q_l - eta * Q_g

    # In accordance with DIN V 18599-10 Tabelle 4
    # Q_tw = max(8.5, 16.5 - 0.05*component_size["Area"]) * 1000
    Q_tw = 12.5 * component_size["Area"]

#    Q_h_spec = Q_h / A_n
    Q_htw_spec = (Q_h + Q_tw) / component_size["Area"]

    # %% Primary Energy

    # Expenditure factor for heating system

    # eg = 0.97  # DIN V 4701-10 Table C.3-4b "Brennwertkessel Verbessert"
    # DIN V 18599-1 does not use Aufwandszahl rather primary energy factors

    f_p_gas = 1.1

    # Q_p = eg * Q_htw_spec * component_size["Area"] # old calculation
    Q_p = f_p_gas * Q_htw_spec * component_size["Area"]

    reference_building = {}
    reference_building["H_t_spec"] = H_t_spec
    reference_building["Q_p"] = Q_p / 1000  # Switch to MWh
    reference_building["Q_s"] = Q_s
    reference_building["Q_i"] = Q_i
    reference_building["Q_tw"] = Q_tw
    reference_building["H_v"] = H_v
    reference_building["f_ql"] = f_ql
    reference_building["eta"] = eta

    reference_building["U-values"] = {}
    reference_building["U-values"]["GroundFloor"] = u_ref["GroundFloor"]
    reference_building["U-values"]["Window"] = u_ref["Window"]
    reference_building["U-values"]["Rooftop"] = u_ref["Rooftop"]
    reference_building["U-values"]["OuterWall"] = u_ref["OuterWall"]

    return reference_building

def reference_building_angepasst(building, grdtgszhl):
    """
    Angepasst an gewählte Parameter für Bedarfsberechnungsmethoden (basierend auf der Tabula Datenbank und gewählte Geometrie für Refernzgebäude

    Parameters
    ----------
    building : dictionary
        Information about the dimensions of the building

    Returns
    -------
    reference_building : dictionary
        Informations about the primary energy requirement and the transmission
        heat losses of the reference building


    """
    #Geometrie
    h1=2.6
    h2=1
    w1=3.92
    w2=2.28
    wroof1=2.21
    wroof2=3.36
    l1=3.3
    l2=2.44
    l3=1.33
    l4=3.3
    hroof=np.sqrt(wroof2**2-w2**2)
    lges=l1+l2+l3+l4

    area = 2 * lges * (2 * w1 + w2)
    volume = lges * 2 * w1 * ((h1+h2) + 0.5 * (h1 - h2 + hroof))
    outerwall = 2 * (2*w1 + lges) * (h1 + h2) + (h1 - h2 + hroof) * 2 * w1
    rooftop = (wroof1 + wroof2) * lges * 2
    groundfloor = 2 * w1 * lges

    window_south = 2 * 1.73 + 3.46 + 8.44
    window_north = 1.4 + 1.73
    window_east = 1.73 * 3
    window_west = 4 * 1.73
    window = window_south + window_north + window_east + window_west

    # %% Define Parameters:

    component_size = {}

    component_size["Area"] = area
    component_size["Volume"] = volume
    component_size["OuterWall"] = outerwall
    component_size["Rooftop"] = rooftop
    component_size["GroundFloor"] = groundfloor
    component_size["Window"] = window

    component_size["Window_south"] = window_south
    component_size["Window_north"] = window_north
    component_size["Window_east"] = window_east
    component_size["Window_west"] = window_west

    # Calculation for one heating period
    t = 185  # days

    # U-Values for EnEV reference building (W/m²K)
    # checked: GEG 2020 Anlage 1
    u_ref = {}
    u_ref["Window"] = 3.2
    u_ref["OuterWall"] = 0.5
    u_ref["GroundFloor"] = 0.51
    u_ref["Rooftop"] = 0.4

    Fx = {}
    Fx["Window"] = 1
    Fx["OuterWall"] = 1
    Fx["GroundFloor"] = 0.8
    Fx["Rooftop"] = 1

    # %% Starting Calculation

    # %% Losses

    # Transmisison heating losses
    building_parts = ("Window", "OuterWall", "Rooftop", "GroundFloor")
    H_t = sum(component_size[i] * u_ref[i] * Fx[i]
              for i in building_parts)  # W/K

    # Surcharge für thermal bridge
    # SOLVED: q.content
    # according to DIN V 4108-6 5.5.2.2. b) U_wb should be 0.1
    # --> checks in accordance with GEG Anlage 1 - 2
    U_wb = 0.05
    H_t = H_t + U_wb * sum(component_size[i] for i in building_parts)  # W/K

    # Specific Transmissison losses
    H_t_spec = H_t / sum(component_size[i] for i in building_parts)  # W/m²K

    # Ventilation losses
    # Still valid for GEG 2020
    # n could be 0.6 but 0.7 is assessment to safe side
    ro_cp = 0.34  # Wh/m³K
    n = 0.7  # 1/h
    H_v = ro_cp * n * component_size["Volume"]  # W/K

    # Total heating losses
    # G_t = 2900  # Kd - Gradtagzahl
    G_t = grdtgszhl
    f_na = 0.95  # Parameter for switching off the heater in the night
    f_ql = 0.024 * G_t * f_na
    Q_l = (H_t + H_v) * f_ql  # kWh

    # %%Profits

    # Annual solar radiation per direction
    I_s = 270  # kWh/m²a
    I_e = 155  # kWh/m²a
    I_w = 155  # kWh/m²a
    I_n = 100  # kWh/m²a

    # Solar profits
    F_f = 0.7  # reduction because of window frame
    F_s = 0.9  # shading coefficient
    F_c = 1.0  # reduction because of sun protection
    g = 0.6 * 0.9  # reduction because of not-vertical radiation

    Q_s = F_f * F_s * F_c * g * (I_s * component_size["Window_south"] +
                                 I_w * component_size["Window_west"] +
                                 I_e * component_size["Window_east"] +
                                 I_n * component_size["Window_north"])
    # kWh/HP - Solar gains per heating period

    # Internal profits
    q_i = 5  # W/m²
    Q_i = 0.024 * q_i * component_size["Area"] * t  # kWh/a

    Q_g = Q_i + Q_s  # kWh/HP - Internal gains per heating period

    # %% Total heating demand

    eta = 0.95  # utilisation factor for heating gains
    Q_h = Q_l - eta * Q_g

    # In accordance with DIN V 18599-10 Tabelle 4
    # Q_tw = max(8.5, 16.5 - 0.05*component_size["Area"]) * 1000
    Q_tw = 12.5 * component_size["Area"]

#    Q_h_spec = Q_h / A_n
    Q_htw_spec = (Q_h + Q_tw) / component_size["Area"]

    # %% Primary Energy

    # Expenditure factor for heating system

    # eg = 0.97  # DIN V 4701-10 Table C.3-4b "Brennwertkessel Verbessert"
    # DIN V 18599-1 does not use Aufwandszahl rather primary energy factors

    f_p_gas = 1.1

    # Q_p = eg * Q_htw_spec * component_size["Area"] # old calculation
    Q_p = f_p_gas * Q_htw_spec * component_size["Area"]

    reference_building = {}
    reference_building["H_t_spec"] = H_t_spec
    reference_building["Q_p"] = Q_p / 1000  # Switch to MWh
    reference_building["Q_s"] = Q_s
    reference_building["Q_i"] = Q_i
    reference_building["Q_tw"] = Q_tw
    reference_building["H_v"] = H_v
    reference_building["f_ql"] = f_ql
    reference_building["eta"] = eta

    reference_building["U-values"] = {}
    reference_building["U-values"]["GroundFloor"] = u_ref["GroundFloor"]
    reference_building["U-values"]["Window"] = u_ref["Window"]
    reference_building["U-values"]["Rooftop"] = u_ref["Rooftop"]
    reference_building["U-values"]["OuterWall"] = u_ref["OuterWall"]

    return reference_building


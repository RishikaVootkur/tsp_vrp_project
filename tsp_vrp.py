import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from collections import defaultdict
import requests
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os
import json

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP and VRP")
        self.root.geometry("1300x800")
        
        # OpenRouteService API key
        self.ors_api_key = "5b3ce3597851110001cf6248427da3d20b5a4b75ac7e92e78ce2c1e7"  # Replace with your actual API key
        
        # European city bounds (restricted to central Europe only)
        self.min_lon = 2.0    # Western Central Europe
        self.max_lon = 20.0   # Eastern Central Europe
        self.min_lat = 45.0   # Southern Central Europe
        self.max_lat = 53.0   # Northern Central Europe
        
        # Default algorithm parameters
        # Simulated Annealing parameters - increased iterations and adjusted cooling
        self.sa_params = {
            "temperature": 1000.0,  # Higher initial temperature
            "cooling_rate": 0.9995,  # Slower cooling
            "iterations": 100000,   # More iterations
            "neighbor_method": "swap"
        }
        
        # ACO TSP parameters - increased iterations and better balance
        self.aco_tsp_params = {
            "n_ants": 30,           # More ants
            "n_iterations": 200,     # More iterations
            "decay": 0.9,           # Faster pheromone decay
            "alpha": 1.0,
            "beta": 3.0,            # Higher importance to distance
            "initial_pheromone": 0.1
        }
        
        # Genetic Algorithm parameters - increased population and generations
        self.ga_params = {
            "population_size": 100,  # Larger population
            "generations": 300,      # More generations
            "mutation_rate": 0.2,    # Higher mutation rate
            "crossover_rate": 0.8,
            "num_vehicles": 5,
            "selection_method": "tournament"
        }
        
        # ACO VRP parameters - increased iterations and adjusted parameters
        self.aco_vrp_params = {
            "n_ants": 30,           # More ants
            "n_iterations": 200,     # More iterations
            "decay": 0.9,           # Faster pheromone decay
            "alpha": 1.0,
            "beta": 3.0,            # Higher importance to distance
            "num_vehicles": 5,
            "initial_pheromone": 0.1
        }
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.tsp_tab = ttk.Frame(self.notebook)
        self.vrp_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tsp_tab, text="Traveler Salesman Problem")
        self.notebook.add(self.vrp_tab, text="Vehicle Routing Problem")
        
        # Initialize TSP tab
        self.setup_tsp_tab()
        
        # Initialize VRP tab
        self.setup_vrp_tab()
        
        # Cities data
        self.cities = []
        self.lat_lon_cities = []  # Store actual lat/lon for map integration
        self.city_names = []     # Store city names
        self.distances = []
        self.num_cities = 100    # Default is now 100 cities
        self.depot_index = 0     # For VRP
        
        # Solution storage
        self.sa_solution = None
        self.aco_tsp_solution = None
        self.ga_solution = None
        self.aco_vrp_solution = None
        
        # For API usage tracking (to avoid exceeding limits)
        self.api_calls = 0
        self.max_api_calls = 40  # Limit API calls to avoid exceeding free tier
        
        # Continental European cities only with good road connectivity
        self.major_cities = [
            # Central France (inland only)
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Lyon", "lat": 45.7640, "lon": 4.8357},
            {"name": "Dijon", "lat": 47.3220, "lon": 5.0415},
            {"name": "Clermont-Ferrand", "lat": 45.7772, "lon": 3.0869},
            {"name": "Limoges", "lat": 45.8315, "lon": 1.2578},
            {"name": "Orleans", "lat": 47.9029, "lon": 1.9039},
            {"name": "Reims", "lat": 49.2577, "lon": 4.0319},
            
            # Benelux (inland only)
            {"name": "Brussels", "lat": 50.8503, "lon": 4.3517},
            {"name": "Liege", "lat": 50.6326, "lon": 5.5797},
            {"name": "Namur", "lat": 50.4673, "lon": 4.8719},
            {"name": "Luxembourg", "lat": 49.6116, "lon": 6.1319},
            
            # Switzerland
            {"name": "Zurich", "lat": 47.3769, "lon": 8.5417},
            {"name": "Geneva", "lat": 46.2044, "lon": 6.1432},
            {"name": "Bern", "lat": 46.9480, "lon": 7.4474},
            {"name": "Basel", "lat": 47.5596, "lon": 7.5886},
            {"name": "Lausanne", "lat": 46.5197, "lon": 6.6323},
            {"name": "Lucerne", "lat": 47.0502, "lon": 8.3093},
            
            # Germany (central and south)
            {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
            {"name": "Munich", "lat": 48.1351, "lon": 11.5820},
            {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
            {"name": "Stuttgart", "lat": 48.7758, "lon": 9.1829},
            {"name": "Cologne", "lat": 50.9375, "lon": 6.9603},
            {"name": "Leipzig", "lat": 51.3397, "lon": 12.3731},
            {"name": "Dresden", "lat": 51.0504, "lon": 13.7373},
            {"name": "Nuremberg", "lat": 49.4521, "lon": 11.0767},
            {"name": "Hannover", "lat": 52.3759, "lon": 9.7320},
            {"name": "Dusseldorf", "lat": 51.2277, "lon": 6.7735},
            {"name": "Dortmund", "lat": 51.5136, "lon": 7.4653},
            {"name": "Essen", "lat": 51.4556, "lon": 7.0116},
            {"name": "Bonn", "lat": 50.7374, "lon": 7.0982},
            {"name": "Mannheim", "lat": 49.4875, "lon": 8.4660},
            {"name": "Karlsruhe", "lat": 49.0069, "lon": 8.4037},
            {"name": "Heidelberg", "lat": 49.3988, "lon": 8.6724},
            {"name": "Freiburg", "lat": 47.9990, "lon": 7.8421},
            {"name": "Augsburg", "lat": 48.3705, "lon": 10.8978},
            {"name": "Regensburg", "lat": 49.0134, "lon": 12.1016},
            {"name": "Würzburg", "lat": 49.7913, "lon": 9.9534},
            {"name": "Erfurt", "lat": 50.9847, "lon": 11.0299},
            {"name": "Jena", "lat": 50.9272, "lon": 11.5864},
            {"name": "Magdeburg", "lat": 52.1205, "lon": 11.6276},
            {"name": "Kassel", "lat": 51.3127, "lon": 9.4797},
            
            # Austria
            {"name": "Vienna", "lat": 48.2082, "lon": 16.3738},
            {"name": "Graz", "lat": 47.0707, "lon": 15.4395},
            {"name": "Linz", "lat": 48.3059, "lon": 14.2863},
            {"name": "Salzburg", "lat": 47.8095, "lon": 13.0550},
            {"name": "Innsbruck", "lat": 47.2692, "lon": 11.4041},
            {"name": "Klagenfurt", "lat": 46.6228, "lon": 14.3051},
            
            # Czech Republic
            {"name": "Prague", "lat": 50.0755, "lon": 14.4378},
            {"name": "Brno", "lat": 49.1951, "lon": 16.6068},
            {"name": "Ostrava", "lat": 49.8209, "lon": 18.2625},
            {"name": "Plzen", "lat": 49.7384, "lon": 13.3736},
            {"name": "Olomouc", "lat": 49.5955, "lon": 17.2582},
            {"name": "Liberec", "lat": 50.7663, "lon": 15.0543},
            
            # Slovakia
            {"name": "Bratislava", "lat": 48.1486, "lon": 17.1077},
            {"name": "Kosice", "lat": 48.7164, "lon": 21.2611},
            {"name": "Zilina", "lat": 49.2231, "lon": 18.7394},
            {"name": "Banska Bystrica", "lat": 48.7395, "lon": 19.1536},
            
            # Hungary
            {"name": "Budapest", "lat": 47.4979, "lon": 19.0402},
            {"name": "Debrecen", "lat": 47.5316, "lon": 21.6273},
            {"name": "Szeged", "lat": 46.2530, "lon": 20.1414},
            {"name": "Miskolc", "lat": 48.1035, "lon": 20.7784},
            {"name": "Pecs", "lat": 46.0727, "lon": 18.2323},
            
            # Poland (central)
            {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122},
            {"name": "Krakow", "lat": 50.0647, "lon": 19.9450},
            {"name": "Lodz", "lat": 51.7592, "lon": 19.4560},
            {"name": "Wroclaw", "lat": 51.1079, "lon": 17.0385},
            {"name": "Poznan", "lat": 52.4064, "lon": 16.9252},
            {"name": "Katowice", "lat": 50.2598, "lon": 19.0215},
            {"name": "Lublin", "lat": 51.2465, "lon": 22.5684},
            
            # Slovenia
            {"name": "Ljubljana", "lat": 46.0569, "lon": 14.5058},
            {"name": "Maribor", "lat": 46.5547, "lon": 15.6467}
        ]
        
        # List of excluded locations (will be ignored even if they appear in major_cities)
        self.excluded_locations = [
            # All coastal cities
            "Amsterdam", "Rotterdam", "The Hague", "Gdansk", "Hamburg", "Bremen",
            "Copenhagen", "Malmo", "Gothenburg", "Stockholm", "Oslo", "Helsinki",
            "Tallinn", "Riga", "Marseille", "Nice", "Genoa", "Venice", "Naples",
            "Barcelona", "Valencia", "Lisbon", "Porto", "Bilbao", "Palma",
            
            # All Russian cities
            "Moscow", "Saint Petersburg", "Kaliningrad", "Nizhny Novgorod", "Kazan",
            "Samara", "Rostov-on-Don", "Voronezh", "Krasnodar", "Volgograd",
            
            # Other excluded areas
            "Kiev", "Minsk", "Bucharest", "Sofia", "Athens", "Istanbul"
        ]
        
    def setup_tsp_tab(self):
        # Create frames for algorithms
        self.tsp_frame = tk.Frame(self.tsp_tab)
        self.tsp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create panels for each algorithm
        self.sa_frame = tk.Frame(self.tsp_frame, bg="navy", bd=2, relief=tk.RAISED)
        self.sa_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.aco_tsp_frame = tk.Frame(self.tsp_frame, bg="navy", bd=2, relief=tk.RAISED)
        self.aco_tsp_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.tsp_frame.grid_columnconfigure(0, weight=1)
        self.tsp_frame.grid_columnconfigure(1, weight=1)
        self.tsp_frame.grid_rowconfigure(0, weight=1)
        
        # Set up SA frame
        sa_title = tk.Label(self.sa_frame, text="Simulated Annealing", font=("Arial", 20, "bold"), bg="navy", fg="white")
        sa_title.pack(pady=5)
        
        self.sa_fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.sa_canvas = FigureCanvasTkAgg(self.sa_fig, self.sa_frame)
        self.sa_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.sa_ax = self.sa_fig.add_subplot(111)
        
        # Metrics for SA
        self.sa_metrics_frame = tk.Frame(self.sa_frame, bg="navy")
        self.sa_metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(self.sa_metrics_frame, text="Execution time", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=0, padx=5)
        self.sa_time_var = tk.StringVar(value="0 ms")
        tk.Label(self.sa_metrics_frame, textvariable=self.sa_time_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=1, padx=5)
        
        tk.Label(self.sa_metrics_frame, text="Cost of found solution", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=2, padx=5)
        self.sa_cost_var = tk.StringVar(value="0")
        tk.Label(self.sa_metrics_frame, textvariable=self.sa_cost_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=3, padx=5)
        
        # Map buttons
        self.sa_map_btn = tk.Button(self.sa_metrics_frame, text="See on Map", command=lambda: self.show_on_map("sa"))
        self.sa_map_btn.grid(row=0, column=4, padx=10)
        
        # Set up ACO frame
        aco_title = tk.Label(self.aco_tsp_frame, text="Ant Colony", font=("Arial", 20, "bold"), bg="navy", fg="white")
        aco_title.pack(pady=5)
        
        self.aco_tsp_fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.aco_tsp_canvas = FigureCanvasTkAgg(self.aco_tsp_fig, self.aco_tsp_frame)
        self.aco_tsp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.aco_tsp_ax = self.aco_tsp_fig.add_subplot(111)
        
        # Metrics for ACO
        self.aco_tsp_metrics_frame = tk.Frame(self.aco_tsp_frame, bg="navy")
        self.aco_tsp_metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(self.aco_tsp_metrics_frame, text="Execution time", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=0, padx=5)
        self.aco_tsp_time_var = tk.StringVar(value="0 ms")
        tk.Label(self.aco_tsp_metrics_frame, textvariable=self.aco_tsp_time_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=1, padx=5)
        
        tk.Label(self.aco_tsp_metrics_frame, text="Cost of found solution", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=2, padx=5)
        self.aco_tsp_cost_var = tk.StringVar(value="0")
        tk.Label(self.aco_tsp_metrics_frame, textvariable=self.aco_tsp_cost_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=3, padx=5)
        
        # Map buttons
        self.aco_map_btn = tk.Button(self.aco_tsp_metrics_frame, text="See on Map", command=lambda: self.show_on_map("aco_tsp"))
        self.aco_map_btn.grid(row=0, column=4, padx=10)
        
        # Controls frame at the bottom
        controls_frame = tk.Frame(self.tsp_tab)
        controls_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Left side (presenters)
        presenters_frame = tk.Frame(controls_frame)
        presenters_frame.grid(row=0, column=0, sticky="w")
        
        tk.Label(presenters_frame, text="Presented by:").grid(row=0, column=0, sticky="w")
        tk.Label(presenters_frame, text="Rishika Reddy Vootkur", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w")
        tk.Label(presenters_frame, text="Jeanne Boucher", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w")
        
        # Center frame for algorithm parameter buttons
        params_frame = tk.Frame(controls_frame)
        params_frame.grid(row=0, column=1)
        
        # Add title
        tk.Label(params_frame, text="Algorithm Parameters", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Add parameter buttons
        buttons_frame = tk.Frame(params_frame)
        buttons_frame.pack()
        
        # SA parameters button
        sa_params_btn = tk.Button(buttons_frame, text="SA Parameters", 
                                 command=lambda: self.show_algorithm_params("sa"))
        sa_params_btn.grid(row=0, column=0, padx=10, pady=5)
        
        # ACO parameters button
        aco_params_btn = tk.Button(buttons_frame, text="ACO Parameters", 
                                  command=lambda: self.show_algorithm_params("aco_tsp"))
        aco_params_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Right side (controls)
        right_frame = tk.Frame(controls_frame)
        right_frame.grid(row=0, column=2, sticky="e")
        
        tk.Label(right_frame, text="Number of cities").grid(row=0, column=0, padx=5)
        
        city_values = [100, 150, 200, 250, 300]  # Minimum 100 cities
        self.city_combobox = ttk.Combobox(right_frame, values=city_values, width=10)
        self.city_combobox.current(0)  # Default to 100 cities
        self.city_combobox.grid(row=0, column=1, padx=5)
        
        generate_btn = tk.Button(right_frame, text="GENERATE", command=self.generate_cities)
        generate_btn.grid(row=0, column=2, padx=5)
        
        solve_btn = tk.Button(right_frame, text="SOLVE", command=self.solve_tsp)
        solve_btn.grid(row=0, column=3, padx=5)
        
        # Configure grid weights
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=2)
        controls_frame.grid_columnconfigure(2, weight=1)
        
    def setup_vrp_tab(self):
        # Create frames for algorithms
        self.vrp_frame = tk.Frame(self.vrp_tab)
        self.vrp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create panels for each algorithm
        self.ga_frame = tk.Frame(self.vrp_frame, bg="navy", bd=2, relief=tk.RAISED)
        self.ga_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.aco_vrp_frame = tk.Frame(self.vrp_frame, bg="navy", bd=2, relief=tk.RAISED)
        self.aco_vrp_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.vrp_frame.grid_columnconfigure(0, weight=1)
        self.vrp_frame.grid_columnconfigure(1, weight=1)
        self.vrp_frame.grid_rowconfigure(0, weight=1)
        
        # Set up GA frame
        ga_title = tk.Label(self.ga_frame, text="Genetic Algorithm", font=("Arial", 20, "bold"), bg="navy", fg="white")
        ga_title.pack(pady=5)
        
        self.ga_fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.ga_canvas = FigureCanvasTkAgg(self.ga_fig, self.ga_frame)
        self.ga_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.ga_ax = self.ga_fig.add_subplot(111)
        
        # Metrics for GA
        self.ga_metrics_frame = tk.Frame(self.ga_frame, bg="navy")
        self.ga_metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(self.ga_metrics_frame, text="Execution time", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=0, padx=5)
        self.ga_time_var = tk.StringVar(value="0 ms")
        tk.Label(self.ga_metrics_frame, textvariable=self.ga_time_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=1, padx=5)
        
        tk.Label(self.ga_metrics_frame, text="Cost of found solution", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=2, padx=5)
        self.ga_cost_var = tk.StringVar(value="0")
        tk.Label(self.ga_metrics_frame, textvariable=self.ga_cost_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=3, padx=5)
        
        # Map buttons
        self.ga_map_btn = tk.Button(self.ga_metrics_frame, text="See on Map", command=lambda: self.show_on_map("ga"))
        self.ga_map_btn.grid(row=0, column=4, padx=10)
        
        # Set up ACO frame for VRP
        aco_title = tk.Label(self.aco_vrp_frame, text="Ant Colony", font=("Arial", 20, "bold"), bg="navy", fg="white")
        aco_title.pack(pady=5)
        
        self.aco_vrp_fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.aco_vrp_canvas = FigureCanvasTkAgg(self.aco_vrp_fig, self.aco_vrp_frame)
        self.aco_vrp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.aco_vrp_ax = self.aco_vrp_fig.add_subplot(111)
        
        # Metrics for ACO in VRP
        self.aco_vrp_metrics_frame = tk.Frame(self.aco_vrp_frame, bg="navy")
        self.aco_vrp_metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(self.aco_vrp_metrics_frame, text="Execution time", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=0, padx=5)
        self.aco_vrp_time_var = tk.StringVar(value="0 ms")
        tk.Label(self.aco_vrp_metrics_frame, textvariable=self.aco_vrp_time_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=1, padx=5)
        
        tk.Label(self.aco_vrp_metrics_frame, text="Cost of found solution", font=("Arial", 10), bg="navy", fg="white").grid(row=0, column=2, padx=5)
        self.aco_vrp_cost_var = tk.StringVar(value="0")
        tk.Label(self.aco_vrp_metrics_frame, textvariable=self.aco_vrp_cost_var, font=("Arial", 10, "bold"), bg="navy", fg="white").grid(row=0, column=3, padx=5)
        
        # Map buttons
        self.aco_vrp_map_btn = tk.Button(self.aco_vrp_metrics_frame, text="See on Map", command=lambda: self.show_on_map("aco_vrp"))
        self.aco_vrp_map_btn.grid(row=0, column=4, padx=10)
        
        # Controls frame at the bottom
        controls_frame = tk.Frame(self.vrp_tab)
        controls_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Left side (presenters)
        presenters_frame = tk.Frame(controls_frame)
        presenters_frame.grid(row=0, column=0, sticky="w")
        
        tk.Label(presenters_frame, text="Presented by:").grid(row=0, column=0, sticky="w")
        tk.Label(presenters_frame, text="Rishika Reddy Vootkur", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w")
        tk.Label(presenters_frame, text="Jeanne Boucher", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w")
        
        # Center frame for algorithm parameter buttons
        params_frame = tk.Frame(controls_frame)
        params_frame.grid(row=0, column=1)
        
        # Add title
        tk.Label(params_frame, text="Algorithm Parameters", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Add parameter buttons
        buttons_frame = tk.Frame(params_frame)
        buttons_frame.pack()
        
        # GA parameters button
        ga_params_btn = tk.Button(buttons_frame, text="GA Parameters", 
                                command=lambda: self.show_algorithm_params("ga"))
        ga_params_btn.grid(row=0, column=0, padx=10, pady=5)
        
        # ACO VRP parameters button
        aco_vrp_params_btn = tk.Button(buttons_frame, text="ACO Parameters", 
                                     command=lambda: self.show_algorithm_params("aco_vrp"))
        aco_vrp_params_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Right side (controls)
        right_frame = tk.Frame(controls_frame)
        right_frame.grid(row=0, column=2, sticky="e")
        
        tk.Label(right_frame, text="Number of cities").grid(row=0, column=0, padx=5)
        
        city_values = [100, 150, 200, 250]  # Minimum 100 cities
        self.vrp_city_combobox = ttk.Combobox(right_frame, values=city_values, width=10)
        self.vrp_city_combobox.current(0)  # Default to 100 cities
        self.vrp_city_combobox.grid(row=0, column=1, padx=5)
        
        generate_btn = tk.Button(right_frame, text="GENERATE", command=self.generate_cities_vrp)
        generate_btn.grid(row=0, column=2, padx=5)
        
        solve_btn = tk.Button(right_frame, text="SOLVE", command=self.solve_vrp)
        solve_btn.grid(row=0, column=3, padx=5)
        
        # Button to change depot city
        change_depot_btn = tk.Button(right_frame, text="CHANGE DEPOT", command=self.change_depot_city)
        change_depot_btn.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Configure grid weights
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=2)
        controls_frame.grid_columnconfigure(2, weight=1)
    
    def generate_realistic_european_cities(self, n):
        """Generate n random cities from real European city locations"""
        base_cities = []
        
        # Filter out excluded locations that would cause water routing issues
        available_cities = [city for city in self.major_cities 
                           if city["name"] not in self.excluded_locations]
        
        if n <= len(available_cities):
            # If we have enough base cities, sample without replacement
            base_cities = random.sample(available_cities, k=n)
        else:
            # If we need more cities than in our list, use all cities and then create variations
            base_cities = available_cities.copy()
            
            # For the remaining cities, create variations of existing ones
            remaining = n - len(base_cities)
            # Create additional city variations
            for i in range(remaining):
                # Pick a random city as base
                base_city = random.choice(available_cities)
                
                # Create a variation with a slightly different location (within 15-30km)
                lat_offset = random.uniform(-0.15, 0.15)  # ~15km in lat direction
                lon_offset = random.uniform(-0.2, 0.2)  # ~15km in lon direction
                
                # Add variation to base city
                variation = {
                    "name": f"{base_city['name']} Area {i+1}",
                    "lat": base_city["lat"] + lat_offset,
                    "lon": base_city["lon"] + lon_offset
                }
                
                base_cities.append(variation)
        
        # Extract lat/lon and names
        lat_lon_cities = []
        city_names = []
        
        for city in base_cities:
            lat_lon_cities.append((city["lat"], city["lon"]))
            city_names.append(city["name"])
            
        # Add small random variation to all coordinates to ensure uniqueness
        for i in range(len(lat_lon_cities)):
            lat, lon = lat_lon_cities[i]
            # Add small random offset (±2-3km)
            lat_offset = random.uniform(-0.02, 0.02)  # ~2km in lat direction
            lon_offset = random.uniform(-0.03, 0.03)  # ~2km in lon direction
            lat_lon_cities[i] = (lat + lat_offset, lon + lon_offset)
        
        return lat_lon_cities, city_names
    
    def generate_cities(self):
        self.num_cities = int(self.city_combobox.get())
        
        # Generate realistic city locations
        self.lat_lon_cities, self.city_names = self.generate_realistic_european_cities(self.num_cities)
        
        # Create scaled coordinates for visualization (0-100 scale)
        self.cities = []
        for lat, lon in self.lat_lon_cities:
            x = ((lon - self.min_lon) / (self.max_lon - self.min_lon)) * 100
            y = ((lat - self.min_lat) / (self.max_lat - self.min_lat)) * 100
            self.cities.append((x, y))
        
        # Calculate distance matrix
        self.distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.distances[i][j] = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                                                  (self.cities[i][1] - self.cities[j][1])**2)
        
        # Clear previous plots
        self.sa_ax.clear()
        self.aco_tsp_ax.clear()
        
        # Plot cities with labels
        for i, city in enumerate(self.cities):
            self.sa_ax.plot(city[0], city[1], 'ko')
            # Add city name for larger datasets
            if self.num_cities <= 20:  # Only show labels for small datasets
                self.sa_ax.annotate(f"{i}:{self.city_names[i]}", 
                                   (city[0], city[1]), 
                                   xytext=(5, 0), 
                                   textcoords='offset points',
                                   fontsize=8)
            else:
                self.sa_ax.annotate(f"{i}", 
                                   (city[0], city[1]), 
                                   xytext=(5, 0), 
                                   textcoords='offset points',
                                   fontsize=8)
            
            self.aco_tsp_ax.plot(city[0], city[1], 'ko')
            # Copy the same labeling for ACO plot
            if self.num_cities <= 20:
                self.aco_tsp_ax.annotate(f"{i}:{self.city_names[i]}", 
                                        (city[0], city[1]), 
                                        xytext=(5, 0), 
                                        textcoords='offset points',
                                        fontsize=8)
            else:
                self.aco_tsp_ax.annotate(f"{i}", 
                                        (city[0], city[1]), 
                                        xytext=(5, 0), 
                                        textcoords='offset points',
                                        fontsize=8)
        
        self.sa_ax.set_xlim(-5, 105)
        self.sa_ax.set_ylim(-5, 105)
        self.aco_tsp_ax.set_xlim(-5, 105)
        self.aco_tsp_ax.set_ylim(-5, 105)
        
        self.sa_canvas.draw()
        self.aco_tsp_canvas.draw()
        
        # Reset solutions
        self.sa_solution = None
        self.aco_tsp_solution = None
        
        # Reset API call counter
        self.api_calls = 0
    
    def generate_cities_vrp(self):
        self.num_cities = int(self.vrp_city_combobox.get())
        
        # Generate realistic city locations
        self.lat_lon_cities, self.city_names = self.generate_realistic_european_cities(self.num_cities)
        
        # Create scaled coordinates for visualization (0-100 scale)
        self.cities = []
        for lat, lon in self.lat_lon_cities:
            x = ((lon - self.min_lon) / (self.max_lon - self.min_lon)) * 100
            y = ((lat - self.min_lat) / (self.max_lat - self.min_lat)) * 100
            self.cities.append((x, y))
        
        # Default depot is the first city
        self.depot_index = 0
        
        # Calculate distance matrix
        self.distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.distances[i][j] = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                                                  (self.cities[i][1] - self.cities[j][1])**2)
        
        # Clear previous plots
        self.ga_ax.clear()
        self.aco_vrp_ax.clear()
        
        # Plot cities without displaying city names
        for i, city in enumerate(self.cities):
            if i == self.depot_index:
                # Mark depot with a different color
                self.ga_ax.plot(city[0], city[1], 'ro', markersize=10)
                self.ga_ax.annotate("DEPOT", 
                                  (city[0], city[1]), 
                                  xytext=(5, 0), 
                                  textcoords='offset points',
                                  fontsize=8,
                                  weight='bold')
                
                self.aco_vrp_ax.plot(city[0], city[1], 'ro', markersize=10)
                self.aco_vrp_ax.annotate("DEPOT", 
                                       (city[0], city[1]), 
                                       xytext=(5, 0), 
                                       textcoords='offset points',
                                       fontsize=8,
                                       weight='bold')
            else:
                self.ga_ax.plot(city[0], city[1], 'ko')
                self.aco_vrp_ax.plot(city[0], city[1], 'ko')
        
        self.ga_ax.set_xlim(-5, 105)
        self.ga_ax.set_ylim(-5, 105)
        self.aco_vrp_ax.set_xlim(-5, 105)
        self.aco_vrp_ax.set_ylim(-5, 105)
        
        self.ga_canvas.draw()
        self.aco_vrp_canvas.draw()
        
        # Reset solutions
        self.ga_solution = None
        self.aco_vrp_solution = None
        
        # Reset API call counter
        self.api_calls = 0
    
    def change_depot_city(self):
        """Allow user to select a new depot city from the existing cities"""
        if not self.cities or self.num_cities <= 1:
            tk.messagebox.showinfo("Error", "Please generate cities first")
            return
        
        # Create a more user-friendly interface for selecting the depot city
        depot_window = tk.Toplevel(self.root)
        depot_window.title("Select Depot City")
        depot_window.geometry("300x400")
        
        tk.Label(depot_window, text="Select a city to use as depot:", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Create a listbox with all cities
        listbox = tk.Listbox(depot_window, width=40, height=15)
        listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Add cities to listbox without displaying the full names which can be long
        for i in range(len(self.city_names)):
            # Truncate long city names
            city_name = self.city_names[i]
            if len(city_name) > 20:
                city_name = city_name[:17] + "..."
            listbox.insert(tk.END, f"City {i}: {city_name}")
        
        # Highlight current depot
        listbox.selection_set(self.depot_index)
        listbox.see(self.depot_index)
        
        # Create button to confirm selection
        def on_select():
            selection = listbox.curselection()
            if selection:
                self.depot_index = selection[0]
                self.update_depot_display()
                depot_window.destroy()
            else:
                tk.messagebox.showinfo("Error", "Please select a city")
                
        tk.Button(depot_window, text="Set as Depot", command=on_select).pack(pady=10)
    
    def update_depot_display(self):
        """Update the display to show the new depot"""
        # Update GA plot
        self.ga_ax.clear()
        for i, city in enumerate(self.cities):
            if i == self.depot_index:
                self.ga_ax.plot(city[0], city[1], 'ro', markersize=10)
                self.ga_ax.annotate("DEPOT", 
                                  (city[0], city[1]), 
                                  xytext=(5, 0), 
                                  textcoords='offset points',
                                  fontsize=8,
                                  weight='bold')
            else:
                self.ga_ax.plot(city[0], city[1], 'ko')
                # No city names displayed
        
        # Update ACO plot
        self.aco_vrp_ax.clear()
        for i, city in enumerate(self.cities):
            if i == self.depot_index:
                self.aco_vrp_ax.plot(city[0], city[1], 'ro', markersize=10)
                self.aco_vrp_ax.annotate("DEPOT", 
                                       (city[0], city[1]), 
                                       xytext=(5, 0), 
                                       textcoords='offset points',
                                       fontsize=8,
                                       weight='bold')
            else:
                self.aco_vrp_ax.plot(city[0], city[1], 'ko')
                # No city names displayed
        
        self.ga_ax.set_xlim(-5, 105)
        self.ga_ax.set_ylim(-5, 105)
        self.aco_vrp_ax.set_xlim(-5, 105)
        self.aco_vrp_ax.set_ylim(-5, 105)
        
        self.ga_canvas.draw()
        self.aco_vrp_canvas.draw()
        
        # Reset solutions since the depot changed
        self.ga_solution = None
        self.aco_vrp_solution = None
    
    def show_on_map(self, algorithm):
        """Display the solution on a real map using Folium"""
        if not self.lat_lon_cities:
            tk.messagebox.showinfo("Error", "Please generate cities first")
            return
        
        # Check if solution exists
        if algorithm == "sa" and self.sa_solution is None:
            tk.messagebox.showinfo("Error", "Please solve TSP with Simulated Annealing first")
            return
        elif algorithm == "aco_tsp" and self.aco_tsp_solution is None:
            tk.messagebox.showinfo("Error", "Please solve TSP with Ant Colony first")
            return
        elif algorithm == "ga" and self.ga_solution is None:
            tk.messagebox.showinfo("Error", "Please solve VRP with Genetic Algorithm first")
            return
        elif algorithm == "aco_vrp" and self.aco_vrp_solution is None:
            tk.messagebox.showinfo("Error", "Please solve VRP with Ant Colony first")
            return
            
        # Reset API counter
        self.api_calls = 0
            
        # Create a map centered at the average location
        avg_lat = sum(lat for lat, _ in self.lat_lon_cities) / len(self.lat_lon_cities)
        avg_lon = sum(lon for _, lon in self.lat_lon_cities) / len(self.lat_lon_cities)
        
        # Create the map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, 
                     tiles='OpenStreetMap')
        
        # Create a marker cluster for cities
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for cities with proper icons
        for i, (lat, lon) in enumerate(self.lat_lon_cities):
            popup_content = f"<b>City {i}: {self.city_names[i]}</b>"
            
            if (algorithm == "ga" or algorithm == "aco_vrp") and i == self.depot_index:
                # Use depot icon (red)
                folium.Marker(
                    [lat, lon], 
                    popup=folium.Popup(popup_content, max_width=200),
                    tooltip=f"Depot: {self.city_names[i]}",
                    icon=folium.Icon(color='red', icon='home', prefix='fa')
                ).add_to(m)  # Add depot directly to map for visibility
            else:
                # Use regular city icon (blue)
                folium.Marker(
                    [lat, lon], 
                    popup=folium.Popup(popup_content, max_width=200),
                    tooltip=f"City {i}: {self.city_names[i]}",
                    icon=folium.Icon(color='blue', icon='map-marker', prefix='fa')
                ).add_to(marker_cluster)
        
        # Add title
        title_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 400px; height: 45px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:18px; font-weight: bold; padding: 10px;">
                        {title}
            </div>
        '''
        
        if algorithm == "sa":
            m.get_root().html.add_child(folium.Element(
                title_html.format(title=f"TSP Solution - Simulated Annealing ({len(self.sa_solution)} cities)")
            ))
            # Add paths based on algorithm
            self.add_route_to_map(m, self.sa_solution, "red", "TSP Route")
            
        elif algorithm == "aco_tsp":
            m.get_root().html.add_child(folium.Element(
                title_html.format(title=f"TSP Solution - Ant Colony ({len(self.aco_tsp_solution)} cities)")
            ))
            self.add_route_to_map(m, self.aco_tsp_solution, "blue", "TSP Route")
            
        elif algorithm == "ga":
            m.get_root().html.add_child(folium.Element(
                title_html.format(title=f"VRP Solution - Genetic Algorithm ({len(self.lat_lon_cities)} cities)")
            ))
            # For VRP, add multiple routes
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkblue', 'cadetblue', 
                     'darkgreen', 'darkred', 'darkpurple']
            for i, route in enumerate(self.ga_solution):
                vehicle_num = i + 1
                self.add_route_to_map(
                    m, route, 
                    colors[i % len(colors)], 
                    f"Vehicle {vehicle_num} Route"
                )
                
        elif algorithm == "aco_vrp":
            m.get_root().html.add_child(folium.Element(
                title_html.format(title=f"VRP Solution - Ant Colony ({len(self.lat_lon_cities)} cities)")
            ))
            # For VRP, add multiple routes
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkblue', 'cadetblue', 
                     'darkgreen', 'darkred', 'darkpurple']
            for i, route in enumerate(self.aco_vrp_solution):
                vehicle_num = i + 1
                self.add_route_to_map(
                    m, route, 
                    colors[i % len(colors)], 
                    f"Vehicle {vehicle_num} Route"
                )
        
        # Add legend for VRP solutions
        if algorithm in ["ga", "aco_vrp"]:
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; 
                            border:2px solid grey; z-index:9999; font-size:14px;
                            background-color: white; padding: 10px; 
                            opacity: 0.8; border-radius: 6px;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Vehicle Routes:</div>
            '''
            
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkblue', 
                     'cadetblue', 'darkgreen', 'darkred', 'darkpurple']
            routes = self.ga_solution if algorithm == "ga" else self.aco_vrp_solution
            
            for i in range(len(routes)):
                color = colors[i % len(colors)]
                legend_html += f'''
                    <div>
                        <i class="fa fa-truck" style="color:{color}; margin-right:5px;"></i>
                        Vehicle {i+1}
                    </div>
                '''
            
            legend_html += '''</div>'''
            m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add route distance information
        if algorithm == "sa":
            distance = float(self.sa_cost_var.get())
            info_html = f'''
                <div style="position: fixed; bottom: 50px; right: 50px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px; opacity: 0.8; border-radius: 6px;">
                    <div><b>Total Distance:</b> {distance:.2f} units</div>
                    <div><small>Time: {self.sa_time_var.get()}</small></div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(info_html))
        elif algorithm == "aco_tsp":
            distance = float(self.aco_tsp_cost_var.get())
            info_html = f'''
                <div style="position: fixed; bottom: 50px; right: 50px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px; opacity: 0.8; border-radius: 6px;">
                    <div><b>Total Distance:</b> {distance:.2f} units</div>
                    <div><small>Time: {self.aco_tsp_time_var.get()}</small></div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(info_html))
        elif algorithm == "ga":
            distance = float(self.ga_cost_var.get())
            info_html = f'''
                <div style="position: fixed; bottom: 50px; right: 50px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px; opacity: 0.8; border-radius: 6px;">
                    <div><b>Total Distance:</b> {distance:.2f} units</div>
                    <div><small>Time: {self.ga_time_var.get()}</small></div>
                    <div><small>Vehicles: {len(self.ga_solution)}</small></div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(info_html))
        elif algorithm == "aco_vrp":
            distance = float(self.aco_vrp_cost_var.get())
            info_html = f'''
                <div style="position: fixed; bottom: 50px; right: 50px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px; opacity: 0.8; border-radius: 6px;">
                    <div><b>Total Distance:</b> {distance:.2f} units</div>
                    <div><small>Time: {self.aco_vrp_time_var.get()}</small></div>
                    <div><small>Vehicles: {len(self.aco_vrp_solution)}</small></div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(info_html))
        
        # Add Folium plugins for better user experience
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.LocateControl().add_to(m)
        folium.plugins.MeasureControl().add_to(m)
        
        # Save map as HTML file and open in browser
        map_file = f"{algorithm}_map.html"
        m.save(map_file)
        webbrowser.open('file://' + os.path.realpath(map_file))
    
    def add_route_to_map(self, m, route, color, route_name="Route"):
        """Add a route to the map using OpenRouteService directions"""
        # Create a feature group for the route
        route_group = folium.FeatureGroup(name=route_name)
        m.add_child(route_group)
        
        is_vrp = len(route) > 0 and route[0] == route[-1]  # Check if this is a VRP route (starts and ends at depot)
        
        for i in range(len(route) - 1):
            if self.api_calls >= self.max_api_calls:
                # If we've exceeded our API limit, just use direct lines
                self.add_direct_line(route_group, route[i], route[i+1], color)
                continue
                
            start_idx = route[i]
            end_idx = route[i + 1]
            
            start_city = self.lat_lon_cities[start_idx]
            end_city = self.lat_lon_cities[end_idx]
            
            # Check if the connection might cross water (approximate check based on distance)
            lat1, lon1 = start_city
            lat2, lon2 = end_city
            R = 6371  # Earth radius in km
            
            # Calculate great-circle distance
            dLat = math.radians(lat2 - lat1)
            dLon = math.radians(lon2 - lon1)
            a = (math.sin(dLat/2) * math.sin(dLat/2) +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dLon/2) * math.sin(dLon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            # If distance is too large (possibly crossing sea), use a direct line with warning
            if distance > 500:  # Stricter distance limit for central Europe (500km is a reasonable max)
                self.add_direct_line(route_group, start_idx, end_idx, color, "Long distance - may require ferry")
                continue
            
            start_coords = self.lat_lon_cities[start_idx][::-1]  # Convert to [lon, lat] for ORS
            end_coords = self.lat_lon_cities[end_idx][::-1]  # Convert to [lon, lat] for ORS
            
            try:
                # Get directions from OpenRouteService
                url = f"https://api.openrouteservice.org/v2/directions/driving-car"
                headers = {
                    'Authorization': self.ors_api_key,
                    'Content-Type': 'application/json; charset=utf-8'
                }
                
                data = {
                    "coordinates": [start_coords, end_coords],
                    "format": "geojson",
                    "preference": "fastest",  # Use fastest route
                    "instructions": False,    # We don't need turn instructions
                    "avoid_features": ["ferries"]  # Avoid ferries/water crossings
                }
                
                # If API key is not valid or not set, use direct line
                if not self.ors_api_key or "YOUR_API_KEY" in self.ors_api_key:
                    self.add_direct_line(route_group, start_idx, end_idx, color)
                    continue
                
                # Make API request for routing
                self.api_calls += 1  # Count API calls
                response = requests.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    route_data = response.json()
                    
                    # Extract route geometry and distance
                    route_coords = []
                    distance = 0
                    
                    for feature in route_data['features']:
                        if feature['geometry']['type'] == 'LineString':
                            # Convert [lon, lat] to [lat, lon] for folium
                            route_coords = [(coord[1], coord[0]) for coord in feature['geometry']['coordinates']]
                            if 'properties' in feature and 'summary' in feature['properties']:
                                distance = feature['properties']['summary'].get('distance', 0)
                    
                    # Add route line to map with popup showing distance
                    if route_coords:
                        # Check if route distance is much longer than direct distance
                        # This can indicate a water crossing with a long detour
                        direct_distance = self.calculate_direct_distance(start_city, end_city)
                        
                        if distance/1000 > direct_distance * 2 and distance/1000 > direct_distance + 200:
                            # If routed distance is much longer than direct (e.g., routing around water bodies)
                            self.add_direct_line(route_group, start_idx, end_idx, color, 
                                               "No direct road route")
                        else:
                            # Create a popup with the city names and distance
                            popup_text = f"<b>{self.city_names[start_idx]} → {self.city_names[end_idx]}</b><br>"
                            popup_text += f"Distance: {distance/1000:.1f} km"
                            
                            # Add the line with popup
                            folium.PolyLine(
                                route_coords,
                                color=color,
                                weight=4,
                                opacity=0.8,
                                popup=folium.Popup(popup_text, max_width=300),
                                tooltip=f"{self.city_names[start_idx]} → {self.city_names[end_idx]}"
                            ).add_to(route_group)
                    else:
                        # Fallback to direct line if no route found
                        self.add_direct_line(route_group, start_idx, end_idx, color, "No route found")
                else:
                    # Handle API error
                    print(f"API Error: {response.status_code} - {response.text}")
                    self.add_direct_line(route_group, start_idx, end_idx, color, "API error")
                    
                    # Show warning if we're hitting rate limits
                    if response.status_code in [429, 403]:
                        tk.messagebox.showwarning(
                            "API Limit Reached", 
                            "OpenRouteService API rate limit reached. Using direct lines instead."
                        )
                        self.api_calls = self.max_api_calls  # Force direct lines for remaining routes
                        
            except Exception as e:
                print(f"Error getting directions: {e}")
                # Fallback to direct line
                self.add_direct_line(route_group, start_idx, end_idx, color, f"Error: {str(e)[:30]}")
    
    def calculate_direct_distance(self, start_city, end_city):
        """Calculate direct distance between two cities in km"""
        lat1, lon1 = start_city
        lat2, lon2 = end_city
        R = 6371  # Radius of the Earth in km
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = (math.sin(dLat/2) * math.sin(dLat/2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dLon/2) * math.sin(dLon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance
    
    def add_direct_line(self, route_group, start_idx, end_idx, color, reason="Direct route"):
        """Add a straight line between two points (fallback when API fails)"""
        start_city = self.lat_lon_cities[start_idx]
        end_city = self.lat_lon_cities[end_idx]
        
        # Calculate straight-line distance (approximate)
        distance = self.calculate_direct_distance(start_city, end_city)
        
        # Create popup content
        popup_text = f"<b>{self.city_names[start_idx]} → {self.city_names[end_idx]}</b><br>"
        popup_text += f"Direct distance: {distance:.1f} km<br>"
        popup_text += f"<i>Note: {reason}</i>"
        
        # Add a dashed line to indicate this is not a road route
        folium.PolyLine(
            [(start_city[0], start_city[1]), (end_city[0], end_city[1])],
            color=color,
            weight=3,
            opacity=0.7,
            dash_array='5',  # Make line dashed
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{self.city_names[start_idx]} → {self.city_names[end_idx]} (direct)"
        ).add_to(route_group)
    
    def solve_tsp(self):
        """Solve TSP using both SA and ACO"""
        if not self.cities:
            tk.messagebox.showinfo("Error", "Please generate cities first")
            return
            
        # Reset any previous time variables and show "Solving..." indicator
        self.sa_time_var.set("Solving...")
        self.aco_tsp_time_var.set("Solving...")
        self.root.update()  # Force UI update to show "Solving..." message
        
        # Run Simulated Annealing
        self.run_simulated_annealing()
        
        # Run Ant Colony Optimization for TSP
        self.run_aco_tsp()
    
    def solve_vrp(self):
        """Solve VRP using both GA and ACO"""
        if not self.cities:
            tk.messagebox.showinfo("Error", "Please generate cities first")
            return
        
        # Check if depot is selected
        if self.depot_index is None:
            tk.messagebox.showinfo("Error", "Please select a depot city first")
            return
            
        # Reset any previous time variables
        self.ga_time_var.set("Solving...")
        self.aco_vrp_time_var.set("Solving...")
        self.root.update()  # Force UI update to show "Solving..." message
        
        # Run Genetic Algorithm
        self.run_genetic_algorithm()
        
        # Run Ant Colony Optimization for VRP
        self.run_aco_vrp()
    
    def run_simulated_annealing(self):
        """Solve TSP using Simulated Annealing"""
        # Clear previous solution
        self.sa_ax.clear()
        
        # Plot cities
        for i, city in enumerate(self.cities):
            self.sa_ax.plot(city[0], city[1], 'ko')
            # Add city name
            if self.num_cities <= 20:
                self.sa_ax.annotate(f"{i}:{self.city_names[i]}", 
                                   (city[0], city[1]), 
                                   xytext=(5, 0), 
                                   textcoords='offset points',
                                   fontsize=8)
            else:
                self.sa_ax.annotate(f"{i}", 
                                   (city[0], city[1]), 
                                   xytext=(5, 0), 
                                   textcoords='offset points',
                                   fontsize=8)
        
        # Get parameters
        temperature = self.sa_params["temperature"]
        cooling_rate = self.sa_params["cooling_rate"]
        iterations = self.sa_params["iterations"]
        
        # Initial solution: random permutation
        current_solution = list(range(self.num_cities))
        random.shuffle(current_solution)
        current_distance = self.calculate_route_distance(current_solution)
        
        best_solution = current_solution[:]
        best_distance = current_distance
        
        # Start time
        start_time = time.time()
        
        # SA algorithm
        for iteration in range(iterations):
            if iteration % 1000 == 0:  # Update UI every 1000 iterations
                self.root.update_idletasks()
            
            # Create a new solution by swapping two cities
            new_solution = current_solution[:]
            
            # Use different neighbor generation methods
            if self.sa_params["neighbor_method"] == "swap":
                # Swap two random cities
                i, j = random.sample(range(self.num_cities), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            elif self.sa_params["neighbor_method"] == "insert":
                # Remove a city and insert it at a different position
                i = random.randint(0, self.num_cities - 1)
                j = random.randint(0, self.num_cities - 1)
                if i != j:
                    city = new_solution.pop(i)
                    new_solution.insert(j, city)
            else:
                # Default to swap
                i, j = random.sample(range(self.num_cities), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            # Calculate new distance
            new_distance = self.calculate_route_distance(new_solution)
            
            # Decide if we should accept the new solution
            delta = new_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = new_solution
                current_distance = new_distance
                
            # Update best solution if needed
            if new_distance < best_distance:
                best_solution = new_solution[:]
                best_distance = new_distance
            
            # Cool down
            temperature *= cooling_rate
            
            # Stop if temperature is too low (system is frozen)
            if temperature < 0.01:
                break
        
        # End time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update metrics
        self.sa_time_var.set(f"{execution_time:.2f} sec")
        self.sa_cost_var.set(f"{best_distance:.2f}")
        
        # Add the first city to the end to complete the tour
        if len(best_solution) > 0 and best_solution[0] != best_solution[-1]:
            best_solution.append(best_solution[0])
        
        # Plot best solution
        self.plot_tsp_solution(self.sa_ax, best_solution)
        self.sa_canvas.draw()
        
        # Save best solution for map display
        self.sa_solution = best_solution
    
    def run_aco_tsp(self):
        """Solve TSP using Ant Colony Optimization"""
        # Clear previous solution
        self.aco_tsp_ax.clear()
        
        # Plot cities
        for i, city in enumerate(self.cities):
            self.aco_tsp_ax.plot(city[0], city[1], 'ko')
            # Add city name
            if self.num_cities <= 20:
                self.aco_tsp_ax.annotate(f"{i}:{self.city_names[i]}", 
                                        (city[0], city[1]), 
                                        xytext=(5, 0), 
                                        textcoords='offset points',
                                        fontsize=8)
            else:
                self.aco_tsp_ax.annotate(f"{i}", 
                                        (city[0], city[1]), 
                                        xytext=(5, 0), 
                                        textcoords='offset points',
                                        fontsize=8)
        
        # Get parameters
        n_ants = self.aco_tsp_params["n_ants"]
        n_iterations = self.aco_tsp_params["n_iterations"]
        decay = self.aco_tsp_params["decay"]
        alpha = self.aco_tsp_params["alpha"]
        beta = self.aco_tsp_params["beta"]
        initial_pheromone = self.aco_tsp_params["initial_pheromone"]
        
        # Initialize pheromone matrix
        pheromone = np.ones((self.num_cities, self.num_cities)) * initial_pheromone
        
        best_solution = None
        best_distance = float('inf')
        
        # Start time
        start_time = time.time()
        
        # ACO algorithm
        for iteration in range(n_iterations):
            if iteration % 10 == 0:  # Update UI every 10 iterations
                self.root.update_idletasks()
                
            # For each ant
            ant_solutions = []
            ant_distances = []
            
            for ant in range(n_ants):
                # Start at a random city
                current_city = random.randint(0, self.num_cities - 1)
                solution = [current_city]
                visited = set([current_city])
                
                # Visit all cities
                while len(visited) < self.num_cities:
                    unvisited = [city for city in range(self.num_cities) if city not in visited]
                    
                    # Calculate probabilities for next city
                    probabilities = []
                    for city in unvisited:
                        # Probability based on pheromone and distance
                        prob = (pheromone[current_city][city] ** alpha) * ((1.0 / max(0.1, self.distances[current_city][city])) ** beta)
                        probabilities.append(prob)
                    
                    # Normalize probabilities
                    total = sum(probabilities)
                    if total > 0:
                        probabilities = [p / total for p in probabilities]
                        # Choose next city based on probabilities
                        next_city = np.random.choice(unvisited, p=probabilities)
                    else:
                        # If all probabilities are 0, choose randomly
                        next_city = random.choice(unvisited)
                    
                    solution.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                
                # Return to starting city to complete the tour
                solution.append(solution[0])
                
                # Calculate total distance
                distance = self.calculate_route_distance(solution)
                ant_solutions.append(solution)
                ant_distances.append(distance)
                
                # Update best solution if needed
                if distance < best_distance:
                    best_solution = solution[:]
                    best_distance = distance
            
            # Update pheromone levels
            pheromone *= decay
            
            # Add new pheromones from ant trails
            for ant, solution in enumerate(ant_solutions):
                pheromone_to_add = 1.0 / ant_distances[ant]
                for i in range(len(solution) - 1):
                    pheromone[solution[i]][solution[i+1]] += pheromone_to_add
                    pheromone[solution[i+1]][solution[i]] += pheromone_to_add  # Symmetric TSP
            
            # Elitist strategy: add extra pheromone to best solution
            if best_solution:
                elite_pheromone = 2.0 / best_distance
                for i in range(len(best_solution) - 1):
                    pheromone[best_solution[i]][best_solution[i+1]] += elite_pheromone
                    pheromone[best_solution[i+1]][best_solution[i]] += elite_pheromone
        
        # End time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update metrics
        self.aco_tsp_time_var.set(f"{execution_time:.2f} sec")
        self.aco_tsp_cost_var.set(f"{best_distance:.2f}")
        
        # Plot best solution
        self.plot_tsp_solution(self.aco_tsp_ax, best_solution)
        self.aco_tsp_canvas.draw()
        
        # Save best solution for map display
        self.aco_tsp_solution = best_solution[:-1]  # Remove the last city (return to start) for display
    
    def run_genetic_algorithm(self):
        """Solve VRP using Genetic Algorithm"""
        # Clear previous solution
        self.ga_ax.clear()
        
        # Plot cities with depot highlighted
        for i, city in enumerate(self.cities):
            if i == self.depot_index:
                self.ga_ax.plot(city[0], city[1], 'ro', markersize=10)
                self.ga_ax.annotate("DEPOT", 
                                  (city[0], city[1]), 
                                  xytext=(5, 0), 
                                  textcoords='offset points',
                                  fontsize=8,
                                  weight='bold')
            else:
                self.ga_ax.plot(city[0], city[1], 'ko')
                # No city names displayed
        
        # Get parameters
        population_size = self.ga_params["population_size"]
        generations = self.ga_params["generations"]
        mutation_rate = self.ga_params["mutation_rate"]
        crossover_rate = self.ga_params["crossover_rate"]
        num_vehicles = self.ga_params["num_vehicles"]
        
        # Start time
        start_time = time.time()
        
        # Generate initial population
        population = []
        for _ in range(population_size):
            # Create a random chromosome (excluding depot)
            chromosome = list(range(self.num_cities))
            chromosome.remove(self.depot_index)
            random.shuffle(chromosome)
            population.append(chromosome)
        
        best_solution = None
        best_fitness = float('inf')
        
        # GA algorithm
        for generation in range(generations):
            if generation % 10 == 0:  # Update UI every 10 generations
                self.root.update_idletasks()
                
            # Evaluate fitness for each chromosome
            fitness_scores = []
            for chromosome in population:
                # Split chromosome into vehicle routes
                routes = self.split_into_routes(chromosome, num_vehicles)
                # Calculate total distance
                total_distance = 0
                for route in routes:
                    if route:  # Check if route is not empty
                        # Add depot at beginning and end
                        full_route = [self.depot_index] + route + [self.depot_index]
                        total_distance += self.calculate_route_distance(full_route)
                
                fitness_scores.append(total_distance)
            
            # Update best solution if needed
            min_fitness_idx = fitness_scores.index(min(fitness_scores))
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_solution = self.split_into_routes(population[min_fitness_idx], num_vehicles)
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best chromosome
            elite_idx = fitness_scores.index(min(fitness_scores))
            new_population.append(population[elite_idx])
            
            while len(new_population) < population_size:
                # Selection: tournament selection
                if self.ga_params["selection_method"] == "tournament":
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                else:  # Roulette wheel selection
                    parent1 = self.roulette_selection(population, fitness_scores)
                    parent2 = self.roulette_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutation
                if random.random() < mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < mutation_rate:
                    child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            
            population = new_population
        
        # End time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update metrics
        self.ga_time_var.set(f"{execution_time:.2f} sec")
        self.ga_cost_var.set(f"{best_fitness:.2f}")
        
        # Format solution for plotting: add depot to start and end of each route
        best_solution_with_depot = []
        for route in best_solution:
            if route:  # Check if route is not empty
                best_solution_with_depot.append([self.depot_index] + route + [self.depot_index])
        
        # Plot best solution
        self.plot_vrp_solution(self.ga_ax, best_solution_with_depot)
        self.ga_canvas.draw()
        
        # Save best solution for map display
        self.ga_solution = best_solution_with_depot
    
    def run_aco_vrp(self):
        """Solve VRP using Ant Colony Optimization"""
        # Clear previous solution
        self.aco_vrp_ax.clear()
        
        # Plot cities with depot highlighted
        for i, city in enumerate(self.cities):
            if i == self.depot_index:
                self.aco_vrp_ax.plot(city[0], city[1], 'ro', markersize=10)
                self.aco_vrp_ax.annotate("DEPOT", 
                                       (city[0], city[1]), 
                                       xytext=(5, 0), 
                                       textcoords='offset points',
                                       fontsize=8,
                                       weight='bold')
            else:
                self.aco_vrp_ax.plot(city[0], city[1], 'ko')
                # No city names displayed
        
        # Get parameters
        n_ants = self.aco_vrp_params["n_ants"]
        n_iterations = self.aco_vrp_params["n_iterations"]
        decay = self.aco_vrp_params["decay"]
        alpha = self.aco_vrp_params["alpha"]
        beta = self.aco_vrp_params["beta"]
        num_vehicles = self.aco_vrp_params["num_vehicles"]
        initial_pheromone = self.aco_vrp_params["initial_pheromone"]
        
        # Initialize pheromone matrix
        pheromone = np.ones((self.num_cities, self.num_cities)) * initial_pheromone
        
        best_solution = None
        best_distance = float('inf')
        
        # Start time
        start_time = time.time()
        
        # Cities excluding depot
        non_depot_cities = [i for i in range(self.num_cities) if i != self.depot_index]
        
        # ACO algorithm with multiple colony approach
        for iteration in range(n_iterations):
            if iteration % 10 == 0:  # Update UI every 10 iterations
                self.root.update_idletasks()
                
            # For each ant
            ant_solutions = []
            ant_distances = []
            
            for ant in range(n_ants):
                # Create empty routes for each vehicle
                routes = [[] for _ in range(num_vehicles)]
                
                # Make a copy of cities to visit
                cities_to_visit = non_depot_cities.copy()
                
                # Assign cities to vehicles using ACO principles
                for vehicle in range(num_vehicles):
                    if not cities_to_visit:
                        break
                        
                    # Start from depot
                    current_city = self.depot_index
                    
                    # Build a route for this vehicle
                    route = []
                    while cities_to_visit and len(route) < len(non_depot_cities) // num_vehicles + 2:
                        # Calculate probabilities for next city
                        probabilities = []
                        for city in cities_to_visit:
                            # Probability based on pheromone and distance
                            prob = (pheromone[current_city][city] ** alpha) * ((1.0 / max(0.1, self.distances[current_city][city])) ** beta)
                            probabilities.append(prob)
                        
                        # Normalize probabilities
                        total = sum(probabilities)
                        if total > 0:
                            probabilities = [p / total for p in probabilities]
                            # Choose next city based on probabilities
                            city_idx = np.random.choice(range(len(cities_to_visit)), p=probabilities)
                        else:
                            # If all probabilities are 0, choose randomly
                            city_idx = random.randrange(len(cities_to_visit))
                        
                        next_city = cities_to_visit.pop(city_idx)
                        route.append(next_city)
                        current_city = next_city
                    
                    # Save this vehicle's route
                    routes[vehicle] = route
                
                # Assign any remaining cities
                vehicle_index = 0
                while cities_to_visit:
                    routes[vehicle_index % num_vehicles].append(cities_to_visit.pop(0))
                    vehicle_index += 1
                
                # Add depot at start and end of each route
                full_routes = []
                for route in routes:
                    if route:  # Check if route is not empty
                        full_routes.append([self.depot_index] + route + [self.depot_index])
                
                # Calculate total distance
                total_distance = 0
                for route in full_routes:
                    total_distance += self.calculate_route_distance(route)
                
                ant_solutions.append(full_routes)
                ant_distances.append(total_distance)
                
                # Update best solution if needed
                if total_distance < best_distance:
                    best_solution = [r[:] for r in full_routes]
                    best_distance = total_distance
            
            # Update pheromone levels
            pheromone *= decay
            
            # Add new pheromones from ant trails
            for ant, solution in enumerate(ant_solutions):
                pheromone_to_add = 1.0 / ant_distances[ant]
                for route in solution:
                    for i in range(len(route) - 1):
                        pheromone[route[i]][route[i+1]] += pheromone_to_add
                        pheromone[route[i+1]][route[i]] += pheromone_to_add  # Symmetric VRP
            
            # Elitist strategy: add extra pheromone to best solution
            if best_solution:
                elite_pheromone = 2.0 / best_distance
                for route in best_solution:
                    for i in range(len(route) - 1):
                        pheromone[route[i]][route[i+1]] += elite_pheromone
                        pheromone[route[i+1]][route[i]] += elite_pheromone
        
        # End time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update metrics
        self.aco_vrp_time_var.set(f"{execution_time:.2f} sec")
        self.aco_vrp_cost_var.set(f"{best_distance:.2f}")
        
        # Plot best solution
        self.plot_vrp_solution(self.aco_vrp_ax, best_solution)
        self.aco_vrp_canvas.draw()
        
        # Save best solution for map display
        self.aco_vrp_solution = best_solution
    
    def calculate_route_distance(self, route):
        """Calculate the total distance of a route"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i]][route[i+1]]
        return total_distance
    
    def plot_tsp_solution(self, ax, solution):
        """Plot TSP solution on the given axes"""
        # Plot cities
        for i, city in enumerate(self.cities):
            ax.plot(city[0], city[1], 'ko')
            # Add city name
            if self.num_cities <= 20:
                ax.annotate(f"{i}:{self.city_names[i]}", 
                           (city[0], city[1]), 
                           xytext=(5, 0), 
                           textcoords='offset points',
                           fontsize=8)
            else:
                ax.annotate(f"{i}", 
                           (city[0], city[1]), 
                           xytext=(5, 0), 
                           textcoords='offset points',
                           fontsize=8)
        
        # Plot route connections
        for i in range(len(solution) - 1):
            city1 = self.cities[solution[i]]
            city2 = self.cities[solution[i+1]]
            ax.plot([city1[0], city2[0]], [city1[1], city2[1]], 'b-')
            
            # Add direction arrow if we have few cities
            if self.num_cities <= 20:
                # Add an arrow in the middle of the line
                arrow_pos = 0.5  # Position along the line (0.5 = middle)
                mid_x = city1[0] + arrow_pos * (city2[0] - city1[0])
                mid_y = city1[1] + arrow_pos * (city2[1] - city1[1])
                
                # Direction vector
                dx = city2[0] - city1[0]
                dy = city2[1] - city1[1]
                
                # Normalize and scale
                length = np.sqrt(dx**2 + dy**2)
                dx = dx / length * 5  # Scale arrow size
                dy = dy / length * 5
                
                # Add the arrow
                ax.arrow(mid_x - dx/2, mid_y - dy/2, dx, dy, 
                        head_width=1.5, head_length=1.5, fc='blue', ec='blue')
                
                # Add sequence number
                ax.text(mid_x, mid_y, f"{i}", 
                       fontsize=7, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
    
    def plot_vrp_solution(self, ax, routes):
        """Plot VRP solution with multiple routes on given axes"""
        # Plot cities without names - just dots
        for i, city in enumerate(self.cities):
            if i == self.depot_index:
                ax.plot(city[0], city[1], 'ro', markersize=10)
                # Only show depot name
                ax.annotate(f"DEPOT", 
                           (city[0], city[1]), 
                           xytext=(5, 0), 
                           textcoords='offset points',
                           fontsize=8,
                           weight='bold')
            else:
                ax.plot(city[0], city[1], 'ko')
                # No city names displayed
        
        # Colors for different routes
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        # Plot each route with a different color
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            # Add text label for the vehicle
            ax.text(0.02, 0.98 - (0.05 * i), f"Vehicle {i+1}",
                   horizontalalignment='left',
                   verticalalignment='top',
                   transform=ax.transAxes,
                   color=color,
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Plot route connections
            for j in range(len(route) - 1):
                city1 = self.cities[route[j]]
                city2 = self.cities[route[j+1]]
                ax.plot([city1[0], city2[0]], [city1[1], city2[1]], color=color)
                
                # Add direction arrow if we have few cities
                if self.num_cities <= 20:
                    # Add an arrow in the middle of the line
                    arrow_pos = 0.5  # Position along the line (0.5 = middle)
                    mid_x = city1[0] + arrow_pos * (city2[0] - city1[0])
                    mid_y = city1[1] + arrow_pos * (city2[1] - city1[1])
                    
                    # Direction vector
                    dx = city2[0] - city1[0]
                    dy = city2[1] - city1[1]
                    
                    # Normalize and scale
                    length = np.sqrt(dx**2 + dy**2)
                    dx = dx / length * 5  # Scale arrow size
                    dy = dy / length * 5
                    
                    # Add the arrow
                    ax.arrow(mid_x - dx/2, mid_y - dy/2, dx, dy, 
                            head_width=1.5, head_length=1.5, fc=color, ec=color)
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
    
    def split_into_routes(self, chromosome, num_vehicles):
        """Split a chromosome into multiple vehicle routes"""
        route_size = len(chromosome) // num_vehicles
        routes = []
        
        for i in range(num_vehicles):
            start_idx = i * route_size
            end_idx = (i + 1) * route_size if i < num_vehicles - 1 else len(chromosome)
            routes.append(chromosome[start_idx:end_idx])
        
        return routes
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for GA"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx]
    
    def roulette_selection(self, population, fitness_scores):
        """Roulette wheel selection for GA"""
        # Since we want to minimize distance, invert fitness scores
        # Add a small constant to avoid division by zero
        inverted_scores = [1.0 / (score + 0.1) for score in fitness_scores]
        total_fitness = sum(inverted_scores)
        
        # Normalize to get probabilities
        selection_probs = [score / total_fitness for score in inverted_scores]
        
        # Select based on probabilities
        selected_idx = np.random.choice(range(len(population)), p=selection_probs)
        return population[selected_idx]
    
    def crossover(self, parent1, parent2):
        """Order crossover (OX) for permutation encoding"""
        size = len(parent1)
        
        # Choose crossover points
        cxpoint1 = random.randint(0, size - 1)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint1 > cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
        # Create children with empty values
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy crossover segment
        for i in range(cxpoint1, cxpoint2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        
        # Fill remaining positions for child1
        current_idx = (cxpoint2 + 1) % size
        parent_idx = (cxpoint2 + 1) % size
        
        while -1 in child1:
            if parent2[parent_idx] not in child1:
                child1[current_idx] = parent2[parent_idx]
                current_idx = (current_idx + 1) % size
            parent_idx = (parent_idx + 1) % size
        
        # Fill remaining positions for child2
        current_idx = (cxpoint2 + 1) % size
        parent_idx = (cxpoint2 + 1) % size
        
        while -1 in child2:
            if parent1[parent_idx] not in child2:
                child2[current_idx] = parent1[parent_idx]
                current_idx = (current_idx + 1) % size
            parent_idx = (parent_idx + 1) % size
        
        return child1, child2
    
    def mutate(self, chromosome):
        """Swap mutation for permutation encoding"""
        # Select two random positions
        pos1, pos2 = random.sample(range(len(chromosome)), 2)
        
        # Swap values
        chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        
        return chromosome
    
    def show_algorithm_params(self, algorithm):
        """Display a window with editable algorithm parameters"""
        param_window = tk.Toplevel(self.root)
        
        # Set appropriate title based on algorithm
        if algorithm == "sa":
            param_window.title("Simulated Annealing Parameters")
        elif algorithm == "aco_tsp":
            param_window.title("Ant Colony (TSP) Parameters") 
        elif algorithm == "ga":
            param_window.title("Genetic Algorithm Parameters")
        elif algorithm == "aco_vrp":
            param_window.title("Ant Colony (VRP) Parameters")
            
        param_window.geometry("500x600")
        
        # Store parameter variables and widgets for later access
        self.param_vars = {}
        
        # Frame for parameters
        frame = tk.Frame(param_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add title
        title_text = ""
        if algorithm == "sa":
            title_text = "Simulated Annealing Parameters"
        elif algorithm == "aco_tsp":
            title_text = "Ant Colony Optimization (TSP) Parameters"
        elif algorithm == "ga":
            title_text = "Genetic Algorithm Parameters"
        elif algorithm == "aco_vrp":
            title_text = "Ant Colony Optimization (VRP) Parameters"
            
        title_label = tk.Label(frame, text=title_text, font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create a canvas with scrollbar for many parameters
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add parameters based on algorithm type with editable fields
        row = 0
        if algorithm == "sa":
            # Get current parameters
            params = [
                ("Initial Temperature", str(self.sa_params["temperature"]), "Higher values increase the chance of accepting worse solutions early"),
                ("Cooling Rate", str(self.sa_params["cooling_rate"]), "Controls how quickly the temperature decreases (0-1)"),
                ("Iterations", str(self.sa_params["iterations"]), "Maximum number of iterations"),
                ("Neighbor Generation", self.sa_params["neighbor_method"], "Method to generate neighbors (swap/insert)"),
            ]
            
        elif algorithm == "aco_tsp":
            # Get current parameters
            params = [
                ("Number of Ants", str(self.aco_tsp_params["n_ants"]), "Number of ants in each iteration"),
                ("Number of Iterations", str(self.aco_tsp_params["n_iterations"]), "Maximum number of iterations"),
                ("Pheromone Decay Rate", str(self.aco_tsp_params["decay"]), "Rate at which pheromones evaporate (0-1)"),
                ("Alpha (α)", str(self.aco_tsp_params["alpha"]), "Importance of pheromone trails"),
                ("Beta (β)", str(self.aco_tsp_params["beta"]), "Importance of distances"),
                ("Initial Pheromone", str(self.aco_tsp_params["initial_pheromone"]), "Initial pheromone on all edges"),
            ]
            
        elif algorithm == "ga":
            # Get current parameters
            params = [
                ("Population Size", str(self.ga_params["population_size"]), "Number of chromosomes in the population"),
                ("Number of Generations", str(self.ga_params["generations"]), "Maximum number of generations"),
                ("Mutation Rate", str(self.ga_params["mutation_rate"]), "Probability of mutation (0-1)"),
                ("Crossover Rate", str(self.ga_params["crossover_rate"]), "Probability of crossover (0-1)"),
                ("Number of Vehicles", str(self.ga_params["num_vehicles"]), "Number of vehicles for VRP"),
                ("Selection Method", self.ga_params["selection_method"], "Method to select parents (tournament/roulette)"),
            ]
            
        elif algorithm == "aco_vrp":
            # Get current parameters
            params = [
                ("Number of Ants", str(self.aco_vrp_params["n_ants"]), "Number of ants in each iteration"),
                ("Number of Iterations", str(self.aco_vrp_params["n_iterations"]), "Maximum number of iterations"),
                ("Pheromone Decay Rate", str(self.aco_vrp_params["decay"]), "Rate at which pheromones evaporate (0-1)"),
                ("Alpha (α)", str(self.aco_vrp_params["alpha"]), "Importance of pheromone trails"),
                ("Beta (β)", str(self.aco_vrp_params["beta"]), "Importance of distances"),
                ("Number of Vehicles", str(self.aco_vrp_params["num_vehicles"]), "Number of vehicles for VRP"),
                ("Initial Pheromone", str(self.aco_vrp_params["initial_pheromone"]), "Initial pheromone on all edges"),
            ]
            
        # Display parameters with editable fields
        for param, current_value, description in params:
            tk.Label(scrollable_frame, text=param, font=("Arial", 10, "bold"), anchor="w").grid(
                row=row, column=0, sticky="w", padx=5, pady=5)
            
            # Create an entry field with current value
            param_var = tk.StringVar(value=current_value)
            self.param_vars[param] = param_var
            
            entry = tk.Entry(scrollable_frame, textvariable=param_var, width=10)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            
            # Description on the next row
            tk.Label(scrollable_frame, text=description, font=("Arial", 8), wraplength=380, 
                    anchor="w", justify=tk.LEFT).grid(
                row=row+1, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 10))
            
            row += 2
            
        # Add a separator
        ttk.Separator(scrollable_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        # Buttons frame
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        # Apply button
        apply_btn = tk.Button(button_frame, text="Apply", 
                             command=lambda: self.apply_parameters(algorithm))
        apply_btn.pack(side=tk.LEFT, padx=10)
        
        # Close button
        close_btn = tk.Button(button_frame, text="Close", 
                             command=param_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=10)
        
        # Reset button - uses the original parameters to reset
        if algorithm == "sa":
            default_params = {
                "temperature": 1000.0,
                "cooling_rate": 0.9995,
                "iterations": 100000,
                "neighbor_method": "swap"
            }
        elif algorithm == "aco_tsp":
            default_params = {
                "n_ants": 30,
                "n_iterations": 200,
                "decay": 0.9,
                "alpha": 1.0,
                "beta": 3.0,
                "initial_pheromone": 0.1
            }
        elif algorithm == "ga":
            default_params = {
                "population_size": 100,
                "generations": 300,
                "mutation_rate": 0.2,
                "crossover_rate": 0.8,
                "num_vehicles": 5,
                "selection_method": "tournament"
            }
        elif algorithm == "aco_vrp":
            default_params = {
                "n_ants": 30,
                "n_iterations": 200,
                "decay": 0.9,
                "alpha": 1.0,
                "beta": 3.0,
                "num_vehicles": 5,
                "initial_pheromone": 0.1
            }
            
        reset_btn = tk.Button(button_frame, text="Reset to Defaults", 
                             command=lambda: self.reset_parameters(algorithm, default_params))
        reset_btn.pack(side=tk.LEFT, padx=10)
    
    def apply_parameters(self, algorithm):
        """Apply the parameters entered by the user"""
        try:
            # Validate and store parameters based on algorithm type
            if algorithm == "sa":
                self.sa_params = {
                    "temperature": float(self.param_vars["Initial Temperature"].get()),
                    "cooling_rate": float(self.param_vars["Cooling Rate"].get()),
                    "iterations": int(self.param_vars["Iterations"].get()),
                    "neighbor_method": self.param_vars["Neighbor Generation"].get(),
                }
                # Validate param ranges
                if not (0 < self.sa_params["cooling_rate"] < 1):
                    raise ValueError("Cooling Rate must be between 0 and 1")
                if self.sa_params["iterations"] <= 0:
                    raise ValueError("Iterations must be positive")
                
                # Show confirmation
                tk.messagebox.showinfo("Parameters Applied", 
                                      "Simulated Annealing parameters have been updated")
                
            elif algorithm == "aco_tsp":
                self.aco_tsp_params = {
                    "n_ants": int(self.param_vars["Number of Ants"].get()),
                    "n_iterations": int(self.param_vars["Number of Iterations"].get()),
                    "decay": float(self.param_vars["Pheromone Decay Rate"].get()),
                    "alpha": float(self.param_vars["Alpha (α)"].get()),
                    "beta": float(self.param_vars["Beta (β)"].get()),
                    "initial_pheromone": float(self.param_vars["Initial Pheromone"].get()),
                }
                # Validate param ranges
                if not (0 < self.aco_tsp_params["decay"] < 1):
                    raise ValueError("Pheromone Decay Rate must be between 0 and 1")
                if self.aco_tsp_params["n_ants"] <= 0:
                    raise ValueError("Number of Ants must be positive")
                if self.aco_tsp_params["n_iterations"] <= 0:
                    raise ValueError("Number of Iterations must be positive")
                
                # Show confirmation
                tk.messagebox.showinfo("Parameters Applied", 
                                      "Ant Colony Optimization (TSP) parameters have been updated")
                
            elif algorithm == "ga":
                self.ga_params = {
                    "population_size": int(self.param_vars["Population Size"].get()),
                    "generations": int(self.param_vars["Number of Generations"].get()),
                    "mutation_rate": float(self.param_vars["Mutation Rate"].get()),
                    "crossover_rate": float(self.param_vars["Crossover Rate"].get()),
                    "num_vehicles": int(self.param_vars["Number of Vehicles"].get()),
                    "selection_method": self.param_vars["Selection Method"].get(),
                }
                # Validate param ranges
                if not (0 <= self.ga_params["mutation_rate"] <= 1):
                    raise ValueError("Mutation Rate must be between 0 and 1")
                if not (0 <= self.ga_params["crossover_rate"] <= 1):
                    raise ValueError("Crossover Rate must be between 0 and 1")
                if self.ga_params["population_size"] <= 0:
                    raise ValueError("Population Size must be positive")
                if self.ga_params["generations"] <= 0:
                    raise ValueError("Number of Generations must be positive")
                if self.ga_params["num_vehicles"] <= 0:
                    raise ValueError("Number of Vehicles must be positive")
                
                # Show confirmation
                tk.messagebox.showinfo("Parameters Applied", 
                                      "Genetic Algorithm parameters have been updated")
                
            elif algorithm == "aco_vrp":
                self.aco_vrp_params = {
                    "n_ants": int(self.param_vars["Number of Ants"].get()),
                    "n_iterations": int(self.param_vars["Number of Iterations"].get()),
                    "decay": float(self.param_vars["Pheromone Decay Rate"].get()),
                    "alpha": float(self.param_vars["Alpha (α)"].get()),
                    "beta": float(self.param_vars["Beta (β)"].get()),
                    "num_vehicles": int(self.param_vars["Number of Vehicles"].get()),
                    "initial_pheromone": float(self.param_vars["Initial Pheromone"].get()),
                }
                # Validate param ranges
                if not (0 < self.aco_vrp_params["decay"] < 1):
                    raise ValueError("Pheromone Decay Rate must be between 0 and 1")
                if self.aco_vrp_params["n_ants"] <= 0:
                    raise ValueError("Number of Ants must be positive")
                if self.aco_vrp_params["n_iterations"] <= 0:
                    raise ValueError("Number of Iterations must be positive")
                if self.aco_vrp_params["num_vehicles"] <= 0:
                    raise ValueError("Number of Vehicles must be positive")
                
                # Show confirmation
                tk.messagebox.showinfo("Parameters Applied", 
                                      "Ant Colony Optimization (VRP) parameters have been updated")
        
        except ValueError as e:
            # Show error message
            tk.messagebox.showerror("Invalid Parameters", str(e))
            
        except Exception as e:
            # Show error message for any other exception
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def reset_parameters(self, algorithm, default_params):
        """Reset parameters to default values"""
        if algorithm == "sa":
            self.sa_params = default_params.copy()
            
            # Update entry fields
            self.param_vars["Initial Temperature"].set(str(default_params["temperature"]))
            self.param_vars["Cooling Rate"].set(str(default_params["cooling_rate"]))
            self.param_vars["Iterations"].set(str(default_params["iterations"]))
            self.param_vars["Neighbor Generation"].set(default_params["neighbor_method"])
            
        elif algorithm == "aco_tsp":
            self.aco_tsp_params = default_params.copy()
            
            # Update entry fields
            self.param_vars["Number of Ants"].set(str(default_params["n_ants"]))
            self.param_vars["Number of Iterations"].set(str(default_params["n_iterations"]))
            self.param_vars["Pheromone Decay Rate"].set(str(default_params["decay"]))
            self.param_vars["Alpha (α)"].set(str(default_params["alpha"]))
            self.param_vars["Beta (β)"].set(str(default_params["beta"]))
            self.param_vars["Initial Pheromone"].set(str(default_params["initial_pheromone"]))
            
        elif algorithm == "ga":
            self.ga_params = default_params.copy()
            
            # Update entry fields
            self.param_vars["Population Size"].set(str(default_params["population_size"]))
            self.param_vars["Number of Generations"].set(str(default_params["generations"]))
            self.param_vars["Mutation Rate"].set(str(default_params["mutation_rate"]))
            self.param_vars["Crossover Rate"].set(str(default_params["crossover_rate"]))
            self.param_vars["Number of Vehicles"].set(str(default_params["num_vehicles"]))
            self.param_vars["Selection Method"].set(default_params["selection_method"])
            
        elif algorithm == "aco_vrp":
            self.aco_vrp_params = default_params.copy()
            
            # Update entry fields
            self.param_vars["Number of Ants"].set(str(default_params["n_ants"]))
            self.param_vars["Number of Iterations"].set(str(default_params["n_iterations"]))
            self.param_vars["Pheromone Decay Rate"].set(str(default_params["decay"]))
            self.param_vars["Alpha (α)"].set(str(default_params["alpha"]))
            self.param_vars["Beta (β)"].set(str(default_params["beta"]))
            self.param_vars["Number of Vehicles"].set(str(default_params["num_vehicles"]))
            self.param_vars["Initial Pheromone"].set(str(default_params["initial_pheromone"]))
        
        # Show confirmation
        tk.messagebox.showinfo("Parameters Reset", "Parameters have been reset to default values")

def main():
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
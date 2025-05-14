from typing import Self
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.spatial import ConvexHull
import os

class GT_electricity:
    def __init__(self, days):
        self.days = days
        self.dates = pd.date_range(start='2024-01-01', periods=self.days)
        # Generate hourly timestamps for the simulation
        self.timestamps = pd.date_range(start='2024-01-01 00:00', periods=self.days, freq='H')
        seed = np.random.seed(42)
        self.seed = seed

class simulate_renewable_integration(GT_electricity):

    def __init__(self, days, initial_integration_periods, start_range, duration_range):
        super().__init__(days)
        self.initial_integration_periods = initial_integration_periods
        self.start_range = start_range
        self.duration_range = duration_range

    def simulate_renewable_integration(self):
        """
        Simulates periods of renewable energy integration over the specified number of days.

        Returns:
        pandas.DataFrame: DataFrame with day, date, and renewable integration status.
        """
        # Set seed for reproducibility
        np.random.seed(42)

        # Initialize the integration status array with zeros (no integration)
        integration_status = np.zeros(self.days, dtype=int)

        # Define specific conditions for the initial defined days
        for start, end in self.initial_integration_periods:
            integration_status[start:end] = 1

        # Start simulating more random integration periods for the remaining days
        current_day = self.initial_integration_periods[-1][1]  # Start after the last initial period
        while current_day < self.days:
            # Randomly decide when the next integration period starts
            days_without_integration = np.random.randint(*self.start_range)
            start_of_integration = current_day + days_without_integration

            if start_of_integration >= self.days:
                break

            # Randomly decide the duration of the integration period
            np.random.seed(42)
            integration_duration = np.random.randint(*self.duration_range)
            end_of_integration = min(start_of_integration + integration_duration, self.days)

            # Set the integration status to 1 for the duration of the integration period
            integration_status[start_of_integration:end_of_integration] = 1

            # Move to the day after the current integration ends
            current_day = end_of_integration

        # Generate daily timestamps
        dates = pd.date_range(start='2024-01-01', periods=self.days)

        # Generate hourly timestamps for the simulation
        timestamps = pd.date_range(start='2024-01-01 00:00', periods=self.days * 24, freq='H')

        # Create a DataFrame to store the simulation
        integration_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Renewable Integration': np.repeat(integration_status, 24)
        })

        return integration_df
    

class SimulateElectricityPricesWithRenewables(GT_electricity):
    def __init__(self, days, integration_df, renewable_factor, size1, size2, noise_tuning_factor1, noise_tuning_factor2):
        super().__init__(days)
        self.integration_df = integration_df
        self.renewable_factor = renewable_factor
        self.size1 = size1
        self.size2 = size2
        self.noise_tuning_factor1 = noise_tuning_factor1
        self.noise_tuning_factor2 = noise_tuning_factor2

    def simulate_electricity_prices_with_renewables(self):
        """
        Simulates electricity prices over a given number of points, considering the impact of renewable energy.

        Returns:
        - pd.DataFrame: DataFrame containing the timestamps and simulated electricity prices.
        """
        price_array = np.array([
        0.01393, 0.01317, 0.01146, 0.00998, 0.01017, 0.0107, 0.01246,
        0.01103, 0.00716, 0.01091, 0.0133, 0.01605, 0.02066, 0.02806,
        0.03217, 0.03344, 0.03959, 0.03885, 0.038, 0.04824, 0.04279,
        0.02915, 0.01925, 0.01653, 0.01438, 0.01275, 0.01195, 0.01165,
        0.01181, 0.01336, 0.01741, 0.01516, 0.01393, 0.01465, 0.01798,
        0.02605, 0.04227, 0.05464, 0.04996, 0.05513, 0.05824, 0.0523,
        0.04284, 0.04775, 0.03934, 0.02714, 0.0195, 0.01635, 0.01354,
        0.01266, 0.01233, 0.01092, 0.012, 0.01353, 0.01629, 0.01578,
        0.01414, 0.0158, 0.02183, 0.02803, 0.03321, 0.04493, 0.04775,
        0.06044, 0.07099, 0.05198, 0.04656, 0.08594, 0.08324, 0.03641,
        0.02788, 0.02123
        ])
        # Length of the desired simulation
        simulation_length = 24 * self.days

        # Generate a simulation by repeating and slightly perturbing the given array
        repetitions = simulation_length // len(price_array) + 1
        simulated_prices = np.tile(price_array, repetitions)[:simulation_length]

        # Introduce some random noise to simulate variability
        np.random.seed(42)  # for reproducibility
        noise1 = np.random.normal(0, self.noise_tuning_factor1, size=simulated_prices.shape)
 
        # Add some randomness using sine function
        noise2 = np.sin(np.linspace(0, 2 * np.pi, simulation_length)) * self.noise_tuning_factor2

        # Make noise zero randomly
        noise2[np.random.choice(simulation_length, size=self.size1, replace=False)] = 0
        noise1[np.random.choice(simulation_length, size=self.size2, replace=False)] = 0

        # Correlate prices with renewable data
        integration_arr = self.integration_df['Renewable Integration'].to_numpy()
        simulated_prices += (1 - integration_arr) * self.renewable_factor

        # Combine all noise
        simulated_prices += noise1 + noise2

        # Generate hourly timestamps for the simulation
        timestamps = pd.date_range(start='2024-01-01 00:00', periods=simulation_length, freq='H')

        # Create a DataFrame to store the simulation
        simulated_data = pd.DataFrame({
            'Timestamp': timestamps,
            'Electricity_Price': simulated_prices
        })

        return simulated_data

class Get_electricity_dataframe(GT_electricity):
    def __init__(self, days, BASU):
        super().__init__(days)
        self.BASU = BASU
        self.get_params()
        self.simulate_electricity_prices()

    def get_params(self):
        # Load the dictionary from the JSON file
        with open('config_electricity_prices.json', 'r') as f:
            basu_el_parameters = json.load(f)

        # Convert lists back to tuples for 'initial_integration_periods' if needed
        for basu in basu_el_parameters:
            basu_el_parameters[basu]['initial_integration_periods'] = [
                tuple(period) for period in basu_el_parameters[basu]['initial_integration_periods']
            ]

        self.basu_el_parameters = basu_el_parameters

    def simulate_electricity_prices(self):
        # Use the correctly named attribute and proper indexing
        initial_integration_periods = self.basu_el_parameters[self.BASU]['initial_integration_periods']
        start_range = self.basu_el_parameters[self.BASU]['start_range']
        duration_range = self.basu_el_parameters[self.BASU]['duration_range']
        
        integration_simulator = simulate_renewable_integration(
            self.days,
            initial_integration_periods,
            start_range,
            duration_range
        )
        self.integration_df = integration_simulator.simulate_renewable_integration()

        # Simulate electricity prices with renewable integration
        renewable_factor = self.basu_el_parameters[self.BASU]['renewable_factor']
        size1 = self.basu_el_parameters[self.BASU]['size1']
        size2 = self.basu_el_parameters[self.BASU]['size2']
        noise_tuning_factor1 = self.basu_el_parameters[self.BASU]['noise_tuning_factor1']
        noise_tuning_factor2 = self.basu_el_parameters[self.BASU]['noise_tuning_factor2']
        
        electricity_simulator = SimulateElectricityPricesWithRenewables(
            self.days,
            self.integration_df,
            renewable_factor,
            size1,
            size2,
            noise_tuning_factor1,
            noise_tuning_factor2
        )
        self.simulated_data_df = electricity_simulator.simulate_electricity_prices_with_renewables()

        return self.simulated_data_df
    
    def get_electricity_plot(self):
        """
        Plots the simulated electricity prices over the simulation period.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.simulated_data_df['Timestamp'], self.simulated_data_df['Electricity_Price'])
        plt.xlabel("Timestamp")
        plt.ylabel("Electricity Price")
        plt.title(f"Simulated Electricity Prices for {self.BASU}")
        plt.grid(True)
        plt.show()
    

# Creat Class to simulate the Product Demand
# Load the configuration mapping from a JSON file
# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'asu_config.json')
# Open the configuration file using the computed path
with open(config_path, 'r') as config_file:
    ASU_DATA_FILES = json.load(config_file)

class Get_demand_dataframe:
    def __init__(self, days, basu_name):
        self.days = days
        self.dates = pd.date_range(start='2024-01-01', periods=self.days)
        np.random.seed(42)    # Set the random seed for reproducibility
        
        data_file = ASU_DATA_FILES.get(basu_name)
        if data_file is None:
            raise ValueError(f"Unknown ASU identifier: {basu_name}")
        
        # Load the ASU data from the JSON file
        with open(data_file, 'r') as file:
            self.loaded_data = json.load(file)

        self._initialize_params()
    
    def _initialize_params(self):
        liq_prod_data = self.loaded_data['liq_prod_data']
        # Convert liq_prod_data to a list of tuples for products 'LIN', 'LOX', 'LAR'
        points = [(liq_prod_data['LIN'][i], liq_prod_data['LOX'][i], liq_prod_data['LAR'][i]) 
                  for i in liq_prod_data['LIN']]
        # Compute the convex hull to extract the extreme points
        points_np = np.array(points)
        hull = ConvexHull(points_np)
        self.extreme_points_liqp = points_np[hull.vertices]  # shape: [n_extreme_points, 3]
        self.row_liqprod = self.extreme_points_liqp.shape[0]
    
    def simulate_daily_target_demand(self):
        """
        Simulates daily target demand for the three products ('LIN', 'LOX', 'LAR').
        For each day, 24 hourly production vectors are generated:
          - For each hour, a random lambda vector (of length n_extreme_points) is generated 
            so that its entries sum to 1, and the production is computed as:
                 production_vector = lambda^T * extreme_points_liqp.
          - For each day, a random selection of between 20 and 24 hours is set as active hours.
            In inactive hours, the production is set to a zero vector.
        The hourly production vectors are summed to yield a daily production for each product.
        """
        n_extreme = self.row_liqprod
        daily_demands = []  # list to hold daily production sums (each is a 3-element vector)
        
        for day in range(self.days):
            # Determine active hours (at least 20 out of 24)
            active_hours = np.random.randint(16, 25)        # random number between 20 and 24 (inclusive)
            # Randomly select which hours are active
            active_indices = np.random.choice(24, active_hours, replace=False)
            
            daily_sum = np.zeros(3)
            for hour in range(24):
                if hour in active_indices:
                    # Generate a random lambda vector that sums to 1 (using Dirichlet)
                    lam = np.random.dirichlet(np.ones(n_extreme))
                    # Compute production for the hour
                    production_vector = np.dot(lam, self.extreme_points_liqp)
                else:
                    production_vector = np.zeros(3)
                daily_sum += production_vector
            daily_demands.append(daily_sum)
        
        # Store the daily demands in a DataFrame
        self.daily_demands = pd.DataFrame(
            np.array(daily_demands),
            index=self.dates,
            columns=['LIN', 'LOX', 'LAR']
        )
        return self.daily_demands

    def get_demand_plot(self):
        """
        Plots the daily demand for each product over the simulation period in a single graph.
        """
        if not hasattr(self, 'daily_demands'):
            raise ValueError("Daily demands have not been simulated. Please run simulate_daily_target_demand() first.")
        
        plt.figure(figsize=(12, 6))
        for product in ['LIN', 'LOX', 'LAR']:
            plt.plot(self.daily_demands.index, self.daily_demands[product], label=product)
        plt.xlabel("Date")
        plt.ylabel("Daily Demand")
        plt.title("Daily Demand for Products LIN, LOX, and LAR")
        plt.legend()
        plt.grid(True)
        plt.show()

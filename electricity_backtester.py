

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


class BatteryAsset:


    # note intervals on AEMO are 5 minutes but the some datasets is every 30 minutes such that it matches with the other predictors
    def __init__(self, capacity_mwh=2.0, max_power_mw=1.0, efficiency=0.90, interval_minutes=5):
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.efficiency = efficiency
        self.delta_t = interval_minutes / 60.0
        self.max_delta_e = self.max_power * self.delta_t
        self.soc = 0.0  # Start empty


    def reset(self):
        # sets charge back to 0
        self.soc = 0.0



    def calculate_charge(self, price: float, rate: float = 1.0):
        # ensure rate is positive and bounded between 0.0 and 1.0
        rate = max(0.0, min(1.0, abs(rate)))

        # cannot charge more if already full
        if self.soc >= self.capacity:
            return 0.0, 0.0
        
        # scale maximum interval energy capacity by the strategys target rate
        effective_max_delta_e = self.max_delta_e * rate

        # calculate exact capacity room left in the asset
        space_available = self.capacity - self.soc
        
        # The true physical charge amount is capped by either power capacity or physical space left
        charge_amount = min(effective_max_delta_e, space_available)
        
        # update the state of charge, factoring in thermodynamic round-trip efficiency losses
        self.soc += charge_amount * self.efficiency
        
        # cash flow out (buying power from the wholesale grid market)
        cash_flow = -charge_amount * price

        return charge_amount, cash_flow


    def calculate_discharge(self, price: float, rate: float = 1.0):
        # ensure rate is positive and bounded between 0.0 and 1.0
        rate = max(0.0, min(1.0, float(rate)))

        # if nothing left to discharge, exit early
        if self.soc <= 0.0:
            return 0.0, 0.0
        
        # scale maximum interval energy capacity by the strategy's target rate
        effective_max_delta_e = self.max_delta_e * rate
        
        # energy physically available to pull from the reservoir cells
        energy_available = self.soc
        
        # the true discharge amount is capped by either available energy or maximum physical power rate
        discharge_amount = min(effective_max_delta_e, energy_available)
        
        # remove the energy from the storage vessel
        self.soc -= discharge_amount
        
        # cash flow in (selling injected power back into the wholesale grid market)
        cash_flow = discharge_amount * price

        return discharge_amount, cash_flow
    





class BaseStrategy:

    def __init__(self):
        self.data = None
        
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # override to extract signals, compute indicators, etc.
        return df

    def compute_action(self, idx: int, current_price: float) -> float:
        
        # returns a continuous float between -1.0 and 1.0:
        # -1.0 = max physical CHARGE
        # 0.0 = HOLD
        # +1.0 = max physical DISCHARGE
        # eg. 0.5 = discharge at exactly 50% of max power rate
        return 0.0



class EnergyStorageBacktester:
    # simulation engine that steps through time and handles financial logging

    def __init__(self, df: pl.DataFrame, battery: BatteryAsset, strategy: BaseStrategy):
        self.df = df
        self.battery = battery
        self.strategy = strategy
        

    def run(self):
        # initialise data and prepare features from strategy
        processed_df = self.strategy.prepare_features(self.df)
        
        # convert to fast NumPy arrays to eliminate loop overhead
        rrp_array = processed_df["RRP"].to_numpy()
        target_prices = processed_df["target_next_RRP"].fill_null(0.0).to_numpy()
            
        self.battery.reset()
        
        # storage arrays for diagnostics
        actions = []
        soc_history = []
        pnl_history = []
        

        total_steps = len(processed_df)
        for idx in range(total_steps - 1):

            current_price = rrp_array[idx]
            next_price = target_prices[idx] # what actually cash out at
            
            # query strategy logic (can return a string or a float multiplier)
            intent = self.strategy.compute_action(idx, current_price)

            action = "HOLD"

            cash_flow = 0.0
            
            # evaluating action boundaries along a continuous axis
            if intent < 0:
                # calculate charging scaled by the optimal rate fraction
                charge_rate = abs(intent)
                _, cash_flow = self.battery.calculate_charge(next_price, rate=charge_rate)
                if cash_flow != 0: 
                    action = f"CHARGE_{charge_rate:.2f}" if charge_rate < 1.0 else "CHARGE"
                    
            elif intent > 0:
                # calculating discharging scaled by the optimal rate fraction
                discharge_rate = intent
                _, cash_flow = self.battery.calculate_discharge(next_price, rate=discharge_rate)
                if cash_flow != 0: 
                    action = f"DISCHARGE_{discharge_rate:.2f}" if discharge_rate < 1.0 else "DISCHARGE"
            

            actions.append(action)
            soc_history.append(self.battery.soc)
            pnl_history.append(cash_flow)
            
        # append padding for final row
        actions.append("HOLD")
        soc_history.append(self.battery.soc)
        pnl_history.append(0.0)
        
        # compile performance profile back into Polars
        return processed_df.with_columns([
            pl.Series("Sim_Action", actions),
            pl.Series("Sim_SoC_MWh", soc_history),
            pl.Series("Sim_PnL", pnl_history)
        ])




class BacktestReporter:

    def __init__(self, results_df: pl.DataFrame, battery_capacity_mwh: float):
        
        # tracks cumulative pnl
        self.df = results_df.with_columns([
            pl.col("Sim_PnL").cum_sum().alias("Cumulative_PnL")
        ])
        self.capacity = battery_capacity_mwh

    def print_summary_metrics(self):
        

        total_revenue = self.df["Sim_PnL"].sum()
        
        cum_pnl = self.df["Cumulative_PnL"].to_numpy()
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_drawdown = np.max(drawdowns)
        
        # each action total counts
        charge_events = self.df.filter(pl.col("Sim_Action").str.starts_with("CHARGE")).height
        discharge_events = self.df.filter(pl.col("Sim_Action").str.starts_with("DISCHARGE")).height
        hold_events = self.df.filter(pl.col("Sim_Action") == "HOLD").height
 
        soc_array = self.df["Sim_SoC_MWh"].to_numpy()
        soc_diffs = np.diff(soc_array)
        total_mwh_thru = np.sum(soc_diffs[soc_diffs > 0])
        equivalent_cycles = total_mwh_thru / self.capacity if self.capacity > 0 else 0

        print("=========================================================")
        print("                  PERFORMANCE REPORT                    ")
        print("=========================================================")
        print(f"Net Portfolio Revenue     : ${total_revenue:,.2f}")
        print(f"Maximum Peak-to-Trough DD : ${max_drawdown:,.2f}")
        print(f"Equivalent Full Cycles    : {equivalent_cycles:.2f} cycles")
        print("---------------------------------------------------------")
        print(f"Total Operational Actions :")
        print(f"  - CHARGE Events         : {charge_events}")
        print(f"  - DISCHARGE Events      : {discharge_events}")
        print(f"  - HOLD Events           : {hold_events}")
        print("=========================================================")


    def plot_battery_state(self, num_rows: int = 500):
        
        # shows superimposed spot price (rrp) and battery soc (state of charge) over time
        sample = self.df.head(num_rows)
        
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax2 = ax1.twinx()
        
        ax1.plot(sample["RRP"].to_numpy(), color="gainsboro", label="Spot RRP Price")
        ax1.set_ylabel("Price ($/MWh)", color="gray")
        ax1.set_xlabel("Interval Timesteps")
        
        ax2.plot(sample["Sim_SoC_MWh"].to_numpy(), color="royalblue", alpha=0.8, label="Battery SoC (MWh)")
        ax2.set_ylabel("State of Charge (MWh)", color="royalblue")
        
        plt.title(f"Battery Operational Profile vs Spot Pricing (First {num_rows} Intervals)")
        fig.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()


    def plot_pnl_trajectory(self):
        
        # plots pnl curve over time
        plt.figure(figsize=(14, 4))
        plt.plot(self.df["Cumulative_PnL"].to_numpy(), color="forestgreen", linewidth=2, label="Cumulative PnL")
        plt.axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.title("Cumulative Portfolio Equity Curve")
        plt.xlabel("Interval Timesteps")
        plt.ylabel("Revenue ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_action_price_distribution(self):

        #  categorical breakdown of what prices the strategy chose to act on
        actions = ["CHARGE", "HOLD", "DISCHARGE"]
        data_to_plot = []
        
        for act in actions:
            prices = self.df.filter(pl.col("Sim_Action").str.starts_with(act))["RRP"].to_numpy()
            data_to_plot.append(prices if len(prices) > 0 else [0])
            
        plt.figure(figsize=(10, 5))
        plt.boxplot(data_to_plot, label=actions, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='black'))
        
        plt.title("Price Distribution Profiles Per Operational Action")
        plt.ylabel("Spot Regional Reference Price (RRP)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    

    def plot_underwater(self):

        # plots the drawdown profile over time to analyze risk duration.
        cum_pnl = self.df["Cumulative_PnL"].to_numpy()
        running_max = np.maximum.accumulate(cum_pnl)
        # note drawdown is a negative value representing distance from peak
        drawdowns = cum_pnl - running_max 
        
        plt.figure(figsize=(14, 4))
        plt.fill_between(range(len(drawdowns)), drawdowns, 0, color="crimson", alpha=0.3)
        plt.plot(drawdowns, color="crimson", linewidth=1)
        plt.title("Strategy Drawdown Profile (Underwater Plot)")
        plt.xlabel("Interval Timesteps")
        plt.ylabel("Distance from Equity Peak ($)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    
    def analyse_round_trips(self):

        # pairs charges and discharges into discrete round-trip trades to calculate structural win rate and profit factor.
        actions = self.df["Sim_Action"].to_numpy()
        pnl_array = self.df["Sim_PnL"].to_numpy()
        
        trades = []
        current_trade_pnl = 0
        in_position = False
        
        for i in range(len(actions)):
            if actions[i].startswith("CHARGE"):
                current_trade_pnl += pnl_array[i] # cash outflow/inflow recorded
                in_position = True
            elif actions[i].startswith("DISCHARGE") and in_position:
                current_trade_pnl += pnl_array[i]
                trades.append(current_trade_pnl)
                current_trade_pnl = 0
                in_position = False
                
        trades = np.array(trades)
        if len(trades) == 0:
            print("Not enough completed round-trip trades to analyze.")
            return
            
        gross_profits = np.sum(trades[trades > 0])
        gross_losses = np.abs(np.sum(trades[trades < 0]))
        win_rate = (np.sum(trades > 0) / len(trades)) * 100
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        print("=========================================================")
        print("             ROUND-TRIP TRADE LEDGER                     ")
        print("=========================================================")
        print(f"Total Completed Trades : {len(trades)}")
        print(f"Strategy Win Rate      : {win_rate:.2f}%")
        print(f"Gross Profit Factor    : {profit_factor:.2f}")
        print("=========================================================")



# created to synthetically simulate market
class MarketSimulationSuite:

    def __init__(self, regime_models, transition_matrix, feature_names, default_state_order=[0, 1, 2]):
   
        self.models = regime_models
        self.P = np.array(transition_matrix)
        self.feature_names = feature_names
        self.num_states = self.P.shape[0]
        self.state_order = default_state_order # [Regime_0, Regime_1, Regime_2]


    def simulate_standard_path(self, X_base: np.ndarray, start_regime: int, steps: int) -> tuple:


        # generates a standard markov chain price path scenario using probabilities created from the data
        current_regime = start_regime
        regime_path = [current_regime]
        prices = np.zeros(steps)
        
        for t in range(steps):

            feat_row = X_base[t % len(X_base) : (t % len(X_base)) + 1]
            prices[t] = self.models[current_regime].predict(feat_row)[0]
            
            if t < steps - 1:
                probs = self.P[current_regime, :]
                probs /= np.sum(probs)
                current_regime = np.random.choice(np.arange(self.num_states), p=probs)
                regime_path.append(current_regime)
                
        return prices, np.array(regime_path), np.ones(steps) # note likelihood weights are 1.0 here


    def simulate_rare_event_via_importance_sampling(self, X_base: np.ndarray, start_regime: int, steps: int, target_regime: int, bias_factor: float = 0.40) -> tuple:
        
    
        # forces transitions into a targeted rare state
        # while outputting importance weights to keep backtesting statistics mathematically clean
        # importace sampling
    
        current_regime = start_regime
        regime_path = [current_regime]
        prices = np.zeros(steps)
        weights = np.zeros(steps)
        
        current_weight = 1.0
        
        for t in range(steps):
            feat_row = X_base[t % len(X_base) : (t % len(X_base)) + 1]
            prices[t] = self.models[current_regime].predict(feat_row)[0]
            weights[t] = current_weight
            
            if t < steps - 1:
                p_true = self.P[current_regime, :].copy()
                p_true /= np.sum(p_true)
                
                # create biased proposal distribution Q
                p_biased = np.zeros(self.num_states)
                p_biased[target_regime] = bias_factor
                
                # distribute the remaining probability weight across other states
                remaining_mass = 1.0 - bias_factor
                other_indices = [i for i in range(self.num_states) if i != target_regime]
                
                sum_true_others = np.sum(p_true[other_indices])
                if sum_true_others > 0:
                    for idx in other_indices:
                        p_biased[idx] = (p_true[idx] / sum_true_others) * remaining_mass
                else:
                    p_biased = p_true.copy() # Fallback if no true other path exists
                
                # sample transition using biased distribution Q
                next_regime = np.random.choice(np.arange(self.num_states), p=p_biased)
                
                # update the likelihood ratio weight: W = P(true) / Q(biased)
                likelihood_ratio = p_true[next_regime] / p_biased[next_regime]
                current_weight *= likelihood_ratio
                
                regime_path.append(next_regime)
                current_regime = next_regime
                
        return prices, np.array(regime_path), weights


    def generate_stress_test_dataframe(self, df_template: pl.DataFrame, prices: np.ndarray, weights: np.ndarray) -> pl.DataFrame:

        # packages simulated trajectories directly into a schema prepared for the backtester
        slice_len = len(prices)
        df_sim = df_template.head(slice_len).clone()
        
        return df_sim.with_columns([
            pl.Series("RRP", prices),
            pl.Series("IS_Weight", weights) # can be used to scale pnl calculations inside the backtester
        ])
    



class TimeSeriesWalkForwardEngine:

    def __init__(self, df: pl.DataFrame, initial_train_intervals: int, test_intervals: int, method: str = "rolling"):

        # 'rolling' or 'anchored' walk forward time series testing, training method
        self.df = df
        self.initial_train_len = initial_train_intervals
        self.test_len = test_intervals
        self.method = method.lower()
        
        if self.method not in ["rolling", "anchored"]:
            raise ValueError("Method must be either 'rolling' or 'anchored'")


    def generate_splits(self):

        # yields indices for training and testing slices sequentially across time.

        total_len = self.df.height
        start_train_idx = 0
        end_train_idx = self.initial_train_len
        
        while end_train_idx + self.test_len <= total_len:
            train_slice = (start_train_idx, end_train_idx)
            test_slice = (end_train_idx, end_train_idx + self.test_len)
            
            yield train_slice, test_slice
            
            # Slide the window forward
            if self.method == "rolling":
                start_train_idx += self.test_len
            # If anchored, start_train_idx remains 0
            
            end_train_idx += self.test_len



    def run_walk_forward_backtest(self, pipeline_train_func, strategy_eval_func):

        # executes the walk-forward routine.
        # pipeline_train_func: function that takes a training dataframe slice, trains the markov and GBDT model stack, and returns the models
        # strategy_eval_func: function that takes the trained models, a testing dataframe slice, and runs the backtester, returning a performance metrics dict
    

        all_fold_results = []
        split_generator = self.generate_splits()
        
        for fold, (train_bounds, test_bounds) in enumerate(split_generator):
            print(f"\n--- Processing Walk-Forward Fold {fold} ---")
            print(f"Training Range: Indices {train_bounds[0]} to {train_bounds[1]}")
            print(f"Testing Range : Indices {test_bounds[0]} to {test_bounds[1]}")
            
            df_train = self.df.slice(train_bounds[0], train_bounds[1] - train_bounds[0])
            df_test = self.df.slice(test_bounds[0], test_bounds[1] - test_bounds[0])
            
            # fit the entire regime and machine learning pipeline on this folds' history
            trained_pipeline_assets = pipeline_train_func(df_train)
            
            # then out of sample backtest over the unseen forward horizon
            fold_metrics = strategy_eval_func(trained_pipeline_assets, df_test)
            all_fold_results.append(fold_metrics)
            
        return all_fold_results
    


# example basic implimentation class for a threshold strategy
class PriceThresholdStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()

    def compute_action(self, idx: int, current_price: float) -> float:
        # on the very first row there's no previous price, so hold
        if idx == 0:
            return 0.0 # HOLD

        last_price = self.prices[idx - 1] if hasattr(self, 'prices') else current_price

        if last_price <= 0:
            return -1.0 # CHARGE (max rate)
        elif last_price > 70:
            return 1.0 # DISCHARGE (max rate)
        else:
            return 0.0 # HOLD
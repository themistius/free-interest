from mesa import Model, Agent
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
import numpy as np
import random

def calculate_gini(agents):
    """Calculates the Gini coefficient from a list of household agents' wealth."""
    households = [a for a in agents if isinstance(a, HouseholdAgent)]
    if len(households) < 2: return 0.0
    wealths = sorted([h.wealth for h in households if h.wealth >= 0])
    n = len(wealths)
    if n == 0 or sum(wealths) == 0: return 0.0
    B = sum(i * w for i, w in enumerate(wealths, 1))
    return (2 * B) / (n * sum(wealths)) - (n + 1) / n

class ShockSystem:
    """Handles economic shocks with configurable probabilities and durations."""
    def __init__(self, model):
        self.model = model
        self.active_shocks = {}

    def configure_shocks(self, supply_prob, demand_prob, financial_prob, policy_prob):
        self.shocks = {
            'supply_chain': {
                'prob': supply_prob,
                'duration': 6,
                'effect': lambda: setattr(self.model, 'productivity_shock', 0.95),
                'recovery': lambda: setattr(self.model, 'productivity_shock', 1.0)
            },
            'demand_crisis': {
                'prob': demand_prob,
                'duration': 12,
                'effect': lambda: setattr(self.model, 'consumer_confidence_shock', 0.8),
                'recovery': lambda: setattr(self.model, 'consumer_confidence_shock', 1.0)
            },
            'financial_crisis': {
                'prob': financial_prob,
                'duration': 9,
                'effect': lambda: self.model.bank.implement_credit_crunch(True),
                'recovery': lambda: self.model.bank.implement_credit_crunch(False)
            },
            'policy_uncertainty': {
                'prob': policy_prob,
                'duration': 4,
                'effect': lambda: setattr(self.model, 'policy_uncertainty_shock', 0.8),
                'recovery': lambda: setattr(self.model, 'policy_uncertainty_shock', 1.0)
            }
        }

    def apply_shocks(self):
        for shock_name, params in self.shocks.items():
            if random.random() < params['prob'] and shock_name not in self.active_shocks:
                params['effect']()
                self.active_shocks[shock_name] = params['duration']
        
        for shock_name in list(self.active_shocks.keys()):
            self.active_shocks[shock_name] -= 1
            if self.active_shocks[shock_name] <= 0:
                self.shocks[shock_name]['recovery']()
                del self.active_shocks[shock_name]

class HouseholdAgent(Agent):
    """Represents a household with economic behaviors."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = max(0, np.random.lognormal(mean=8, sigma=1.5))
        self.deposit = self.wealth
        self.consumption_propensity = 0.5 + np.random.rand() * 0.3
        self.skill_level = 0.5 + np.random.rand() * 0.5
        self.employer = None
        self.network = []
        self.confidence = 0.5
        self.risk_tolerance = 0.3
        self.consumption = 0

    @property
    def is_employed(self): return self.employer is not None

    def update_confidence(self):
        if not self.network:
            network_employment_rate = 1.0 if self.is_employed else 0.0
        else:
            network_employment_rate = sum(1 for f in self.network if f.is_employed) / len(self.network)
        self.confidence = (0.9 * self.confidence + 0.1 * network_employment_rate) * self.model.consumer_confidence_shock

    def step_stage_1(self):
        self.update_confidence()
        self.deposit *= (1 + self.model.bank.deposit_rate / 12)
        self.wealth = self.deposit

    def step_stage_2(self):
        income = self.employer.wage if self.is_employed else 0
        self.wealth += income
        
        self.consumption = (income * self.consumption_propensity) + (self.wealth * 0.02 * self.confidence)
        self.wealth = max(0, self.wealth - self.consumption)
        self.deposit = self.wealth

class FirmAgent(Agent):
    """Represents a firm with production, hiring, and investment behaviors."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.capital = max(0, np.random.lognormal(mean=12, sigma=1.8))
        self.productivity = 0.8 + np.random.rand() * 0.4
        self.employees = []
        self.wage = self.model.avg_wage
        self.output = 0
        self.debt = 0
        self.suppliers = []
        self.price = 1.0

    def get_target_employment(self):
        interest_cost_factor = 1 + self.model.bank.lending_rate
        policy_uncertainty = self.model.policy_uncertainty_shock
        return max(0, int(np.floor(
            self.capital / (self.wage * 12 * interest_cost_factor)
        ) * policy_uncertainty))

    def get_supplier_health(self):
        if not self.suppliers: return 1.0
        return np.mean([max(0, s.capital) for s in self.suppliers]) / 1e6

    def step_stage_1(self):
        target_employees = self.get_target_employment()
        
        # Adjust employment
        if len(self.employees) > target_employees:
            for emp in random.sample(self.employees, len(self.employees) - target_employees):
                emp.employer = None
                self.employees.remove(emp)
                
        if len(self.employees) < target_employees:
            unemployed = [h for h in self.model.get_households() if not h.is_employed]
            potential_hires = sorted(unemployed, key=lambda x: x.skill_level, reverse=True)
            for h in potential_hires[:target_employees - len(self.employees)]:
                h.employer = self
                self.employees.append(h)
        
        # Take loans if conditions are favorable
        if self.capital > 0 and self.debt / self.capital < 2.0:
            loan_amount = self.capital * 0.01 * (1 - self.model.bank.lending_rate / 0.1)
            self.debt += self.model.bank.issue_loan(loan_amount)
            self.capital += loan_amount

        # Calculate output
        labor_input = sum(emp.skill_level for emp in self.employees)
        supplier_factor = 0.8 + 0.2 * self.get_supplier_health()
        self.output = max(0, self.productivity * self.model.productivity_shock * 
                         supplier_factor * np.sqrt(max(0, labor_input * self.capital)))
        self.price = max(0.1, (self.wage / self.productivity) * 1.1)

    def step_stage_2(self):
        total_wages = len(self.employees) * self.wage
        interest_payment = self.debt * (self.model.bank.lending_rate / 12)
        self.capital = max(0, self.capital - (total_wages + interest_payment))
        self.debt = max(0, self.debt - interest_payment)
        
        market_share = self.output / self.model.gdp if self.model.gdp > 0 else 0
        revenue = market_share * self.model.total_consumption
        self.capital += revenue

        if self.capital <= 0:  # Bankruptcy
            for emp in self.employees:
                emp.employer = None
            self.model.bank.handle_default(self.debt)
            self.model.schedule.remove(self)
            self.model.num_bankruptcies += 1

class BankAgent(Agent):
    """Central bank handling monetary policy and credit conditions."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.reserves = 1e7
        self.lending_rate = model.interest_rate + 0.02
        self.deposit_rate = model.interest_rate - 0.01
        self.credit_crunch_factor = 1.0

    def update_rates(self):
        self.lending_rate = (self.model.interest_rate + 0.02) * self.credit_crunch_factor
        self.deposit_rate = max(0, self.model.interest_rate - 0.01)

    def issue_loan(self, amount):
        amount = min(amount, self.reserves)
        self.reserves -= amount
        return amount

    def handle_default(self, amount):
        self.reserves = max(0, self.reserves - amount)

    def implement_credit_crunch(self, is_crunch, severity=2.0):
        self.credit_crunch_factor = severity if is_crunch else 1.0

    def step_stage_1(self): 
        self.update_rates()
        
    def step_stage_2(self): 
        pass

class PolicyTools:
    """Government policy toolkit with fiscal and monetary tools."""
    def __init__(self, model):
        self.model = model
        self.quantitative_easing_active = False
        self.qe_amount = 0
        self.fiscal_multiplier = 1.3
        self.ubi_amount = 0
        self.progressive_tax_rate = 0.2
        
    def implement_quantitative_easing(self, amount):
        """Inject money into the economy through asset purchases"""
        amount = max(0, amount)
        self.quantitative_easing_active = True
        self.qe_amount = amount
        self.model.bank.reserves += amount
        
        households = self.model.get_households()
        total_wealth = sum(max(0, h.wealth) for h in households)
        if total_wealth > 0:
            for h in households:
                h.wealth += amount * (max(0, h.wealth) / total_wealth)
                
    def implement_ubi(self, amount):
        """Universal Basic Income distribution"""
        amount = max(0, amount)
        self.ubi_amount = amount
        households = self.model.get_households()
        if households:
            per_household = amount / len(households)
            for h in households:
                h.wealth += per_household
                self.model.gov_debt += per_household
            
    def collect_taxes(self):
        """Progressive taxation collection"""
        households = self.model.get_households()
        if households:
            for h in households:
                tax = max(0, h.wealth) * self.progressive_tax_rate
                h.wealth = max(0, h.wealth - tax)
                self.model.gov_debt = max(0, self.model.gov_debt - tax / len(households))

class HousingMarket:
    """Real estate market with price dynamics."""
    def __init__(self, model):
        self.model = model
        self.house_price_index = 100
        self.rental_yield = 0.04
        self.housing_stock = max(1, len(model.get_households()) * 1.1)
        self.construction_rate = 0.01
        
    def update_prices(self):
        """Adjust housing prices based on economic conditions"""
        try:
            total_wealth = sum(max(0, h.wealth) for h in self.model.get_households())
            demand_factor = total_wealth / (self.housing_stock * max(1, self.house_price_index))
            rate_factor = 1 / (1 + max(0, self.model.interest_rate))
            self.house_price_index *= (0.9 + 0.1 * demand_factor) * rate_factor
            self.construction_rate = 0.01 * (max(0, self.house_price_index) / 100 - 1)
            self.housing_stock *= (1 + self.construction_rate)
        except:
            pass
        
    def get_affordability_index(self):
        """Calculate housing affordability index"""
        try:
            avg_wealth = np.mean([max(0, h.wealth) for h in self.model.get_households()])
            return avg_wealth / (max(1, self.house_price_index) * 0.2)  # 20% down payment
        except:
            return 0

class TradeSystem:
    """International trade system with exchange rates."""
    def __init__(self, model):
        self.model = model
        self.exchange_rate = 1.0
        self.trade_balance = 0
        self.export_demand = 0.2
        self.import_penetration = 0.15
        
    def update_trade(self):
        """Update trade flows and exchange rates"""
        try:
            # Exchange rate based on interest rate differential
            interest_diff = self.model.interest_rate - 0.02  # Foreign rate assumption
            self.exchange_rate = max(0.5, min(2.0, 
                self.exchange_rate * (1 + 0.01 * interest_diff)))
            
            # Trade balance calculation
            exports = max(0, self.model.gdp * self.export_demand)
            imports = max(0, self.model.total_consumption * self.import_penetration)
            self.trade_balance = exports - imports
            
            # GDP impact with sanity checks
            net_trade = (exports - imports) * 0.1
            if abs(net_trade) < 0.3 * self.model.gdp:  # Prevent unrealistic swings
                self.model.gdp += net_trade
        except:
            pass

class EconomicModel(Model):
    """Main economic model integrating all components."""
    def __init__(self, num_households=1000, num_firms=100, scenario=None, **params):
        super().__init__()
        self.schedule = StagedActivation(self, stage_list=["step_stage_1", "step_stage_2"], shuffle=True)
        self.scenario = scenario
        
        # Economic indicators
        self.gdp = 1_000_000
        self.potential_gdp = 1_000_000
        self.inflation_target = params.get('inflation_target', 0.02)
        self.cpi = 100
        self.interest_rate = 0.03
        self.natural_interest_rate = 0.02
        self.avg_wage = 3000
        self.gov_debt = 500_000
        
        # Shock multipliers
        self.productivity_shock = 1.0
        self.consumer_confidence_shock = 1.0
        self.policy_uncertainty_shock = 1.0
        
        # Tracking variables
        self.total_consumption = 0
        self.num_bankruptcies = 0
        
        # Scenario setup
        if self.scenario:
            self.natural_unemployment = self.scenario.get('pre_crisis_unemployment', 0.05)
            self.unemployment_rate = self.scenario.get('pre_crisis_unemployment', 0.05)
            self.interest_rate = self.scenario.get('pre_crisis_interest_rate', 0.03)
        else:
            self.natural_unemployment = 0.05
            self.unemployment_rate = 0.05

        # Create agents
        self.bank = BankAgent("bank", self)
        self.schedule.add(self.bank)
        
        for i in range(num_households):
            self.schedule.add(HouseholdAgent(f"h_{i}", self))
            
        for i in range(num_firms):
            self.schedule.add(FirmAgent(f"f_{i}", self))
            
        # Create networks
        self.create_social_network(params.get('network_density', 0.05))
        self.create_firm_network()
        
        # Initialize systems
        self.shock_system = ShockSystem(self)
        self.shock_system.configure_shocks(
            params.get('supply_shock_prob', 0.02),
            params.get('demand_shock_prob', 0.02),
            params.get('financial_shock_prob', 0.01),
            params.get('policy_shock_prob', 0.01)
        )
        
        self.policy_tools = PolicyTools(self)
        self.housing_market = HousingMarket(self)
        self.trade_system = TradeSystem(self)

        # Data collection setup
        self.datacollector = DataCollector(
            model_reporters={
                "GDP": "gdp",
                "CPI": "cpi",
                "Unemployment": "unemployment_rate",
                "Interest_Rate": "interest_rate",
                "Gini": lambda m: calculate_gini(m.schedule.agents),
                "Active_Firms": lambda m: len(m.get_firms()),
                "Total_Loans": lambda m: sum(f.debt for f in m.get_firms()),
                "Total_Deposits": lambda m: sum(h.deposit for h in m.get_households()),
                "House_Price_Index": lambda m: m.housing_market.house_price_index,
                "Affordability_Index": lambda m: m.housing_market.get_affordability_index(),
                "Trade_Balance": lambda m: m.trade_system.trade_balance,
                "Exchange_Rate": lambda m: m.trade_system.exchange_rate,
                "Gov_Debt_GDP_Ratio": lambda m: m.gov_debt / m.gdp if m.gdp > 0 else 0
            }
        )

    def get_households(self):
        return [a for a in self.schedule.agents if isinstance(a, HouseholdAgent)]
        
    def get_firms(self):
        return [a for a in self.schedule.agents if isinstance(a, FirmAgent)]

    def create_social_network(self, density):
        households = self.get_households()
        for h in households:
            num_friends = max(0, int(density * len(households)))
            possible_friends = [f for f in households if f != h]
            if possible_friends:
                h.network.extend(random.sample(possible_friends, min(num_friends, len(possible_friends))))

    def create_firm_network(self):
        firms = self.get_firms()
        for f in firms:
            possible_suppliers = [s for s in firms if s != f]
            if possible_suppliers:
                f.suppliers.extend(random.sample(possible_suppliers, min(3, len(possible_suppliers))))

    def monetary_policy(self):
        """Taylor rule implementation for interest rates"""
        try:
            inflation = (self.cpi / self.datacollector.model_vars["CPI"][-1] - 1) * 12 \
                if len(self.datacollector.model_vars["CPI"]) > 0 else self.inflation_target
            inflation_gap = inflation - self.inflation_target
            output_gap = (self.gdp - self.potential_gdp) / max(1, self.potential_gdp)
            target_rate = self.natural_interest_rate + inflation + 0.5 * inflation_gap + 0.5 * output_gap
            self.interest_rate = max(0.001, 0.8 * self.interest_rate + 0.2 * target_rate)
        except:
            pass

    def fiscal_policy(self):
        """Automatic stabilizers implementation"""
        try:
            stimulus = -(self.unemployment_rate - self.natural_unemployment) * 0.1 * self.gdp
            if stimulus > 0:
                self.gov_debt += stimulus
                households = self.get_households()
                if households:
                    per_household = stimulus / len(households)
                    for h in households:
                        h.wealth += per_household
        except:
            pass
    
    def step(self):
        """Main model step function"""
        # Handle scenario events
        if self.scenario:
            current_step = self.schedule.steps
            if current_step == self.scenario['crisis_start_month']:
                self.bank.implement_credit_crunch(
                    True, 
                    self.scenario.get('crisis_financial_shock_severity', 2.5)
                )
                self.consumer_confidence_shock = self.scenario.get('crisis_demand_shock_severity', 0.7)
                self.interest_rate = 0.005
            
            if current_step == self.scenario['crisis_start_month'] + self.scenario.get('duration', 36):
                self.bank.implement_credit_crunch(False)
                self.consumer_confidence_shock = 1.0
        else:
            self.shock_system.apply_shocks()

        # Policy implementations
        self.policy_tools.collect_taxes()
        self.monetary_policy()
        self.fiscal_policy()
        
        # Market updates
        self.housing_market.update_prices()
        self.trade_system.update_trade()
        
        # Agent steps
        self.schedule.step()
        
        # Update macroeconomic indicators
        self.gdp = max(0, sum(f.output for f in self.get_firms()))
        self.total_consumption = max(0, sum(h.consumption for h in self.get_households()))
        
        households = self.get_households()
        if households:
            self.unemployment_rate = sum(1 for h in households if not h.is_employed) / len(households)
        
        firms = self.get_firms()
        if firms:
            prices = [f.price for f in firms]
            outputs = [f.output for f in firms]
            if sum(outputs) > 0:
                self.cpi = np.average(prices, weights=outputs)
        
        # Potential GDP growth
        self.potential_gdp *= 1.002
        
        # Collect data
        self.datacollector.collect(self)
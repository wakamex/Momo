# import time
from eth_utils.address import to_canonical_address
from web3 import Web3, HTTPProvider

# from scipy.optimize import fsolve, newton, brentq, bisect
from decimal import Decimal, getcontext

# pylint: disable=too-many-arguments

getcontext().prec = 50  # precision for Decimal objects

RPC_URL = "https://infra.delv.tech/node"
web3 = Web3(HTTPProvider(RPC_URL))
contract_address = to_canonical_address("0x3B02fF1e626Ed7a8fd6eC5299e2C54e1421B626B")
abi_get_pool_info = [
    {
        "inputs": [],
        "stateMutability": "view",
        "type": "function",
        "name": "getPoolInfo",
        "outputs": [
            {
                "internalType": "struct IHyperdrive.PoolInfo",
                "name": "",
                "type": "tuple",
                "components": [
                    {"internalType": "uint256", "name": "shareReserves", "type": "uint256"},
                    {"internalType": "int256", "name": "shareAdjustment", "type": "int256"},
                    {"internalType": "uint256", "name": "bondReserves", "type": "uint256"},
                    {"internalType": "uint256", "name": "lpTotalSupply", "type": "uint256"},
                    {"internalType": "uint256", "name": "sharePrice", "type": "uint256"},
                    {"internalType": "uint256", "name": "longsOutstanding", "type": "uint256"},
                    {"internalType": "uint256", "name": "longAverageMaturityTime", "type": "uint256"},
                    {"internalType": "uint256", "name": "shortsOutstanding", "type": "uint256"},
                    {"internalType": "uint256", "name": "shortAverageMaturityTime", "type": "uint256"},
                    {"internalType": "uint256", "name": "withdrawalSharesReadyToWithdraw", "type": "uint256"},
                    {"internalType": "uint256", "name": "withdrawalSharesProceeds", "type": "uint256"},
                    {"internalType": "uint256", "name": "lpSharePrice", "type": "uint256"},
                    {"internalType": "uint256", "name": "longExposure", "type": "uint256"},
                ],
            }
        ],
    },
]
abi_get_pool_config = [
    {
        "inputs": [],
        "name": "getPoolConfig",
        "outputs": [
            {
                "components": [
                    {"internalType": "contract IERC20", "name": "baseToken", "type": "address"},
                    {"internalType": "uint256", "name": "initialSharePrice", "type": "uint256"},
                    {"internalType": "uint256", "name": "minimumShareReserves", "type": "uint256"},
                    {"internalType": "uint256", "name": "minimumTransactionAmount", "type": "uint256"},
                    {"internalType": "uint256", "name": "positionDuration", "type": "uint256"},
                    {"internalType": "uint256", "name": "checkpointDuration", "type": "uint256"},
                    {"internalType": "uint256", "name": "timeStretch", "type": "uint256"},
                    {"internalType": "address", "name": "governance", "type": "address"},
                    {"internalType": "address", "name": "feeCollector", "type": "address"},
                    {
                        "components": [
                            {"internalType": "uint256", "name": "curve", "type": "uint256"},
                            {"internalType": "uint256", "name": "flat", "type": "uint256"},
                            {"internalType": "uint256", "name": "governance", "type": "uint256"},
                        ],
                        "internalType": "struct IHyperdrive.Fees",
                        "name": "fees",
                        "type": "tuple",
                    },
                    {"internalType": "uint256", "name": "oracleSize", "type": "uint256"},
                    {"internalType": "uint256", "name": "updateGap", "type": "uint256"},
                ],
                "internalType": "struct IHyperdrive.PoolConfig",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


def calc_spot_price(initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch) -> Decimal:
    initial_share_price = Decimal(initial_share_price)
    share_reserves = Decimal(share_reserves) - Decimal(share_adjustment)
    bond_reserves = Decimal(bond_reserves)
    time_stretch = Decimal(time_stretch)
    return (initial_share_price * share_reserves / bond_reserves) ** time_stretch


def calc_apr(share_reserves, share_adjustment, bond_reserves, initial_share_price, position_duration_days, time_stretch):
    share_reserves = Decimal(share_reserves)
    bond_reserves = Decimal(bond_reserves)
    initial_share_price = Decimal(initial_share_price)
    position_duration_days = Decimal(position_duration_days)
    time_stretch = Decimal(time_stretch)
    annualized_time = position_duration_days / Decimal(365)
    spot_price = calc_spot_price(initial_share_price, share_reserves, share_adjustment, bond_reserves, time_stretch)
    return (Decimal(1) - spot_price) / (spot_price * annualized_time)


def get_pool_info():
    contract = web3.eth.contract(address=contract_address, abi=abi_get_pool_info)
    pool_info = contract.functions.getPoolInfo().call()
    pool_info_fields = [output["name"] for output in abi_get_pool_info[0]["outputs"][0]["components"]]
    pool_info_dict = dict(zip(pool_info_fields, pool_info))
    pool_info_dict = {k: v / 1e18 for k, v in pool_info_dict.items()}
    return pool_info_dict


def get_pool_config():
    contract = web3.eth.contract(address=contract_address, abi=abi_get_pool_config)
    pool_config = contract.functions.getPoolConfig().call()
    pool_config_fields = [output["name"] for output in abi_get_pool_config[0]["outputs"][0]["components"]]
    pool_config_dict = dict(zip(pool_config_fields, pool_config))
    pool_config_dict = {k: v / 1e18 if k in ["initialSharePrice", "minimumShareReserves", "timeStretch"] and k != "fees" else v for k, v in pool_config_dict.items()}
    pool_config_dict["fees"] = tuple(fee / 1e18 for fee in pool_config_dict["fees"])
    pool_config_dict["invertedTimeStretch"] = 1 / pool_config_dict["timeStretch"]
    pool_config_dict["positionDuration"] = {
        "Seconds": pool_config_dict["positionDuration"],
        "Minutes": pool_config_dict["positionDuration"] / 60,
        "Hours": pool_config_dict["positionDuration"] / 60 / 60,
        "Days": pool_config_dict["positionDuration"] / 60 / 60 / 24,
        "Weeks": pool_config_dict["positionDuration"] / 60 / 60 / 24 / 7,
        "Years": pool_config_dict["positionDuration"] / 60 / 60 / 24 / 365,
        "Months": pool_config_dict["positionDuration"] / 60 / 60 / 24 / 365 * 12,
    }
    pool_config_dict["curve_fee"], pool_config_dict["flat_fee"], pool_config_dict["governance_fee"] = pool_config_dict["fees"]
    return pool_config_dict


def calc_bond_reserves(share_reserves, share_price, apr, position_duration_years, inverted_time_stretch):
    return share_price * share_reserves * ((Decimal(1) + apr * position_duration_years) ** inverted_time_stretch)


def diff_to_target(bond_reserves, target_apr, share_reserves, share_adjustment, initial_share_price, position_duration_days, time_stretch):
    apr = calc_apr(share_reserves, share_adjustment, bond_reserves, initial_share_price, position_duration_days, time_stretch)
    return apr - target_apr


def get_apr(pool_config: dict, pool_info: dict, bond_override=None) -> Decimal:
    return calc_apr(
        pool_info["shareReserves"],
        pool_info["shareAdjustment"],
        pool_info["bondReserves"] if bond_override is None else bond_override,
        pool_config["initialSharePrice"],
        pool_config["positionDuration"]["Days"],
        pool_config["timeStretch"],
    )


def trade_to(target_apr, pool_info, pool_config) -> Decimal:
    # Convert arguments to Decimal
    pool_info["shareReserves"] = Decimal(pool_info["shareReserves"])
    pool_info["bondReserves"] = Decimal(pool_info["bondReserves"])
    pool_config["initialSharePrice"] = Decimal(pool_config["initialSharePrice"])
    pool_config["positionDuration"]["Days"] = Decimal(pool_config["positionDuration"]["Days"])
    pool_config["positionDuration"]["Years"] = Decimal(pool_config["positionDuration"]["Years"])
    pool_config["timeStretch"] = Decimal(pool_config["timeStretch"])
    pool_config["invertedTimeStretch"] = Decimal(pool_config["invertedTimeStretch"])
    target_apr = Decimal(target_apr)

    # args = (target_apr, pool_info["shareReserves"], pool_config["initialSharePrice"], pool_config["positionDuration"]["Days"], pool_config["timeStretch"])
    current_bonds = pool_info["bondReserves"]
    print(f"current apr = {get_apr(pool_config, pool_info, current_bonds):.15%}")
    target_bonds = calc_bond_reserves(pool_info["shareReserves"], pool_config["initialSharePrice"], target_apr, pool_config["positionDuration"]["Years"], pool_config["invertedTimeStretch"])
    manual_apr = get_apr(pool_config, pool_info, target_bonds)
    print(f"manual apr = {manual_apr:.15%}", f"diff from target = {(manual_apr - target_apr)/target_apr}")
    return (target_bonds - current_bonds) / Decimal(2)


if __name__ == "__main__":
    pool_info = get_pool_info()
    print(f"{pool_info=}")
    pool_config = get_pool_config()
    print(f"{pool_config=}")
    spot_price = calc_spot_price(
        pool_config["initialSharePrice"],
        pool_info["shareReserves"],
        pool_info["shareAdjustment"],
        pool_info["bondReserves"],
        pool_config["timeStretch"],
    )
    print(f"{spot_price=}")
    apr = calc_apr(
        pool_info["shareReserves"],
        pool_info["shareAdjustment"],
        pool_info["bondReserves"],
        pool_config["initialSharePrice"],
        pool_config["positionDuration"]["Days"],
        pool_config["timeStretch"],
    )
    print(f"{apr=}")
    trade_to(0.05, pool_info, pool_config)

"""
Examples of using CompactTree for various use cases.
"""

from compact_tree import CompactTree


def example_basic_usage():
    """Basic creation and access."""
    print("=== Basic Usage ===")
    
    # Create from a nested dict
    data = {
        "users": {
            "alice": "admin",
            "bob": "user",
        },
        "settings": {
            "theme": "dark",
            "language": "en",
        },
        "version": "1.0.0",
    }
    
    tree = CompactTree.from_dict(data)
    
    # Access like a normal dict
    print(f"Alice's role: {tree['users']['alice']}")
    print(f"Theme: {tree['settings']['theme']}")
    print(f"Version: {tree['version']}")
    
    # Check membership
    print(f"'users' in tree: {'users' in tree}")
    print(f"'admin' in tree: {'admin' in tree}")
    print()


def example_serialization():
    """Save and load from disk."""
    print("=== Serialization ===")
    
    data = {
        "countries": {
            "US": {"capital": "Washington", "population": "331M"},
            "UK": {"capital": "London", "population": "67M"},
            "JP": {"capital": "Tokyo", "population": "126M"},
        }
    }
    
    tree = CompactTree.from_dict(data)
    
    # Save to file
    tree.serialize("countries.ctree")
    print("Saved to countries.ctree")
    
    # Load from file
    loaded = CompactTree("countries.ctree")
    print(f"US capital: {loaded['countries']['US']['capital']}")
    print(f"UK population: {loaded['countries']['UK']['population']}")
    print()


def example_to_dict():
    """Convert back to plain dict."""
    print("=== Convert to Dict ===")
    
    data = {
        "fruits": {
            "apple": "red",
            "banana": "yellow",
        },
        "vegetables": {
            "carrot": "orange",
        }
    }
    
    tree = CompactTree.from_dict(data)
    
    # Convert back to plain dict
    plain_dict = tree.to_dict()
    print(f"Type: {type(plain_dict)}")
    print(f"Contents: {plain_dict}")
    print()


def example_large_dataset():
    """Demonstrate space efficiency with repeated keys/values."""
    print("=== Large Dataset with Deduplication ===")
    
    # Simulate configuration with many repeated values
    config = {}
    for i in range(100):
        config[f"service_{i}"] = {
            "status": "active",  # Repeated value
            "region": "us-east-1",  # Repeated value
            "type": "t2.micro",  # Repeated value
            "id": f"instance-{i}",  # Unique value
        }
    
    tree = CompactTree.from_dict(config)
    print(f"Created tree with {len(config)} services")
    print(f"service_0 status: {tree['service_0']['status']}")
    print(f"service_99 type: {tree['service_99']['type']}")
    print("(Keys and values are deduplicated internally)")
    print()


def example_nested_depth():
    """Deep nesting example."""
    print("=== Deep Nesting ===")
    
    data = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "value": "deep!"
                    }
                }
            }
        }
    }
    
    tree = CompactTree.from_dict(data)
    print(f"Deep value: {tree['level1']['level2']['level3']['level4']['value']}")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_serialization()
    example_to_dict()
    example_large_dataset()
    example_nested_depth()
    
    print("✅ All examples completed!")

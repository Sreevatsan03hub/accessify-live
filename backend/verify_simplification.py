from services.text_simplification_service import get_simplifier

def test_simplification():
    simplifier = get_simplifier()
    
    test_cases = [
        "Students, kindly come to the class at 10am.",
        "Basically we utilize this functionality.",
        "We need your assistance to commence the project.",
        "Please facilitate the methodology regarding this component."
    ]
    
    print("\n--- Simplification Test Results ---")
    for text in test_cases:
        simplified = simplifier.simplify(text)
        print(f"\nOriginal:   {text}")
        print(f"Simplified: {simplified}")
        
    # User's failure case
    user_case = "Students, kindly come to the class at 10am. Don't be lazy."
    simplified_user = simplifier.simplify(user_case)
    if "please" in simplified_user.lower():
        print("\n✅ User Case Fixed: 'kindly' -> 'please'")
    else:
        print("\n❌ User Case Failed: 'kindly' not simplified")

if __name__ == "__main__":
    test_simplification()

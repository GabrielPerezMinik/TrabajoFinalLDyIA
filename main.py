import sys
import cm_dbscan as cdbscan
import lm_ltsm as llstm

###############
# PLACEHOLDER #
###############

def run_arimax():
    print("Running ARIMAX model...")
    # Call ARIMAX function here

def run_random_forest():
    print("Running Random Forest Regression model...")
    # Call Random Forest function here

def run_linear_regression():
    print("Running Linear Regression model...")
    # Call Linear Regression function here

def run_kmeans():
    print("Running K-Means clustering...")
    # Call K-Means function here

def run_gmm():
    print("Running Gaussian Mixture Model (GMM) clustering...")
    # Call GMM function here

def run_hierarchical():
    print("Running Hierarchical Clustering...")
    # Call Hierarchical clustering function here

def main():
        print("Welcome to DataAnalyst program")
        clean_data = input("Do you want to execute data cleaning? (Y)Yes (N)No: ").strip().lower()

        match clean_data:
            case "y":
                print("Executing data cleaning...")
                # Call data cleaning function here
            case "n":
                print("Skipping data cleaning...")
            case _:
                print("Invalid option. Proceeding without data cleaning.")

        while True:
            print("\nSelect an option:")
            print("1. Linear models")
            print("2. Clustering models")
            print("3. Exit")

            option = input("Enter your choice: ").strip()

            match option:
                case "1":
                    while True:
                        print("\nSelect a linear model:")
                        print("1. ARIMAX")
                        print("2. LSTM")
                        print("3. Random Forest Regression")
                        print("4. Linear Regression")
                        print("5. Back to main menu")

                        linear_choice = input("Enter your choice: ").strip()

                        match linear_choice:
                            case "1":
                                run_arimax()
                            case "2":
                                llstm.run()
                            case "3":
                                run_random_forest()
                            case "4":
                                run_linear_regression()
                            case "5":
                                break
                            case _:
                                print("Invalid option. Please try again.")

                case "2":
                    while True:
                        print("\nSelect a clustering model:")
                        print("1. K-Means")
                        print("2. Gaussian Mixture Model (GMM)")
                        print("3. DBSCAN")
                        print("4. Hierarchical Clustering")
                        print("5. Back to main menu")

                        cluster_choice = input("Enter your choice: ").strip()

                        match cluster_choice:
                            case "1":
                                run_kmeans()
                            case "2":
                                run_gmm()
                            case "3":
                                cdbscan.run("data/clean_data.csv")
                            case "4":
                                run_hierarchical()
                            case "5":
                                break
                            case _:
                                print("Invalid option. Please try again.")

                case "3":
                    print("Exiting the program.")
                    sys.exit()
                case _:
                    print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()

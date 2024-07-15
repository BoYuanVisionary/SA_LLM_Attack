import pandas as pd
import matplotlib.pyplot as plt

# # Load the CSV files for sa approach
# df_seed_0_gd = pd.read_csv('seed_0_batch_32_GD.csv')
# df_seed_1_gd = pd.read_csv('seed_1_batch_32_GD.csv')
# df_seed_2_gd = pd.read_csv('seed_2_batch_32_GD.csv')
# df_seed_3_gd = pd.read_csv('seed_3_batch_32_GD.csv')

# # Load the CSV files for SA approach
# df_seed_0_sa = pd.read_csv('seed_0_batch_32_SA.csv')
# df_seed_1_sa = pd.read_csv('seed_1_batch_32_SA.csv')
# df_seed_2_sa = pd.read_csv('seed_2_batch_32_SA.csv')
# df_seed_3_sa = pd.read_csv('seed_3_batch_32_SA.csv')

# # Combine the 'min' columns from all dataframes into new dataframes
# min_values_gd = pd.DataFrame({
#     'seed_0': df_seed_0_gd['min'],
#     'seed_1': df_seed_1_gd['min'],
#     'seed_2': df_seed_2_gd['min'],
#     'seed_3': df_seed_3_gd['min']
# })

# min_values_sa = pd.DataFrame({
#     'seed_0': df_seed_0_sa['min'],
#     'seed_1': df_seed_1_sa['min'],
#     'seed_2': df_seed_2_sa['min'],
#     'seed_3': df_seed_3_sa['min']
# })

# # Compute the mean and std of 'min' values across the four files for both GD and SA
# mean_min_values_gd = min_values_gd.mean(axis=1)
# std_min_values_gd = min_values_gd.std(axis=1)

# mean_min_values_sa = min_values_sa.mean(axis=1)
# std_min_values_sa = min_values_sa.std(axis=1)

# # Convert to lists
# mean_min_values_gd_list = mean_min_values_gd.tolist()
# std_min_values_gd_list = std_min_values_gd.tolist()

# mean_min_values_sa_list = mean_min_values_sa.tolist()
# std_min_values_sa_list = std_min_values_sa.tolist()

# # Plot the shaded areas
# plt.figure(figsize=(10, 6))

# # GD approach
# plt.plot(range(1, 21), mean_min_values_gd_list, label='GD', color='red')
# plt.fill_between(range(1, 21), 
#                  [mean - std for mean, std in zip(mean_min_values_gd_list, std_min_values_gd_list)], 
#                  [mean + std for mean, std in zip(mean_min_values_gd_list, std_min_values_gd_list)], 
#                  color='red', alpha=0.3)

# # SA approach
# plt.plot(range(1, 21), mean_min_values_sa_list, label='SA', color='blue')
# plt.fill_between(range(1, 21), 
#                  [mean - std for mean, std in zip(mean_min_values_sa_list, std_min_values_sa_list)], 
#                  [mean + std for mean, std in zip(mean_min_values_sa_list, std_min_values_sa_list)], 
#                  color='blue', alpha=0.3)

# plt.xlabel('Example')
# plt.ylabel('Min Value')
# plt.title('Mean and Standard Deviation of Min Values Across Experiments (GD vs SA)')
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.savefig('comparison.png')
# plt.close()


# plt.figure(figsize=(10, 6))
# # Collect data from files where seed is from 8 to 20
# for length in [5,10,20,50,100]:
#     min_values_sa = pd.DataFrame()

#     for seed in range(4):
#         df_seed_sa = pd.read_csv(f'./experiments2/cooling_0.6_seed_{seed}_batch_128_test_length_{length}_SA.csv')
#         min_values_sa[f'seed_{seed}'] = df_seed_sa['min']
    
#     mean_min_values_sa = min_values_sa.mean(axis=1).tolist()
#     std_min_values_sa = min_values_sa.std(axis=1).tolist()
#     # plt.errorbar(range(len(mean_min_values_sa)), mean_min_values_sa, yerr=std_min_values_sa, fmt='o', label=f'length_{length}', capsize= 5)
#     plt.scatter(range(1,len(mean_min_values_sa)+1), mean_min_values_sa,  label=f'length_{length}')

# plt.xlabel('Index')
# plt.ylabel('Min Value')
# plt.title('Means and Standard Deviations of Min Values')
# plt.legend()
# plt.savefig('./experiments2/compare_new.png')
# plt.show()



min_values_gd = pd.DataFrame()
for seed in range(0, 7):
    df_seed_gd = pd.read_csv(f'./min_values/seed_{seed}_batch_200_GD.csv')
    min_values_gd[f'seed_{seed}'] = df_seed_gd['min']

# Calculate mean and standard deviation of min values across seeds
mean_min_values_gd = min_values_gd.min(axis=1).tolist()


min_values_sa = pd.DataFrame()
for seed in range(0, 7):
    df_seed_sa = pd.read_csv(f'./min_values/cooling_0.6_seed_{seed}_batch_200_test_length_20_SA.csv')
    min_values_sa[f'seed_{seed}'] = df_seed_sa['min']

# Calculate mean and standard deviation of min values across seeds
mean_min_values_sa = min_values_sa.min(axis=1).tolist()


# Plot the results
plt.figure(figsize=(10, 6))


plt.scatter(range(1,len(mean_min_values_sa)+1), mean_min_values_sa,  label=f'SA')
plt.scatter(range(1,len(mean_min_values_gd)+1), mean_min_values_gd,  label=f'GD')

plt.xlabel('Index')
plt.ylabel('Min Value')
plt.title('Means and Standard Deviations of Min Values')
plt.legend()
plt.savefig('./min_values/compare_all.png')
plt.show()

min_values_gd = pd.DataFrame()
for seed in range(0, 7):
    df_seed_gd = pd.read_csv(f'./min_values/seed_{seed}_batch_200_GD.csv')
    min_values_gd[f'seed_{seed}'] = df_seed_gd['min']

min_values_sa = pd.DataFrame()
for seed in range(0, 7):
    df_seed_sa = pd.read_csv(f'./min_values/cooling_0.6_seed_{seed}_batch_200_test_length_20_SA.csv')
    min_values_sa[f'seed_{seed}'] = df_seed_sa['min']

mean_min_values_gd = min_values_gd.min(axis=1)
mean_min_values_sa = min_values_sa.min(axis=1)

filtered_data = (mean_min_values_gd > 0.25) | (mean_min_values_sa > 0.25)

mean_min_values_gd_filtered = mean_min_values_gd[filtered_data]
mean_min_values_sa_filtered = mean_min_values_sa[filtered_data]

# Plot the filtered results
plt.figure(figsize=(10, 6))
plt.scatter(range(1, len(mean_min_values_sa_filtered) + 1), mean_min_values_sa_filtered, label='SA')
plt.scatter(range(1, len(mean_min_values_gd_filtered) + 1), mean_min_values_gd_filtered, label='GD')
plt.xlabel('Index')
plt.ylabel('Min Value')
plt.title('Means of Min Values (Filtered)')
plt.legend()
plt.savefig('./min_values/compare_filtered.png')
plt.show()
import pstats
p = pstats.Stats('profiling/train_profile.txt')
p.sort_stats('cumulative').print_stats(10)
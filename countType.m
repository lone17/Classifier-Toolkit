files = {"RD_1XST.csv"; "RD_2P_P.csv"; "RD_2X.csv"; "RD_3P.csv"; "RD_4P.csv";
         "RD_5P.csv"; "RD_6P_P.csv"; "RD_7P.csv"; "RD_8P.csv"; "RD-1P.csv";
         "RDT_1P.csv"; "RDT_1RX.csv"; "RDT_2P.csv"};

% load data
for j = 1:size(files)
    f = files{j};
    fprintf("\nFile: %s\n", f);
    y = csvread(f)(:, end);

    for i = 0:10
        fprintf("   Number of type %i: %i\n", i, sum(y == i));
    end
end

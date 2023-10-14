# TD_Systemes_Intelligents_Avances

## Loading tensorboard to view graphs

To view the graphs in the MLP using the following command

```bash
tensorboard --logdir /path/to/logs --load_fast true
```

And replace "/path/to/logs" with the output of the writer so here for the MLP for example the command is:

```bash
tensorboard --logdir Results/MLP/ --load_fast true
```
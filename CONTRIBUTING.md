# Adding your contribution to hpnn

First of all, I would like to thank you for taking the time to help us improve our project. There have been a lot of work involved in hpnn, and contributions from the community are very much valued.\
Thank you so much for giving some of your precious time! m(_ _)m

Before you go on with submitting modifications, there are a little guidelines to follow which are aimed to facilitate the inclusion of your proposal.

## Answering issues

If you bring an answer to a BUG, FEATURE, or IMPROVE request from the "issues board", please mention it clearly in your pull request.
In that case, if your modifications are accepted, the issue can be closed as well.

### propose to fix to a BUG

If your contribution fixes a BUG that was not part the "issues board", please open a BUG issue before proposing a fix!\
In the "Additional context" of the BUG issue, please mention that you are working on it and/or already have a solution.\
In addition to fixing the BUG problem (mandatory), the resulting code have to pass a consistency test.\
This means that the code including your modifications will have to pass some test against known results.\
The difference will have to be either negligible or justified.

The goal is to ensure that the code remains solid even after many modifications.

### propose a new FEATURE

Please state precisely what feature is added and why.\
If your contribution does not answer a FEATURE request on the "issues board", there is no need to fill up one.\
If you do fill one up, don't forget to mention in the "Additional context" that you already have a proposal.

Generally speaking, it is better when a feature is prepared in a different git branch first.\
The new FEATURE will have to be carefully check as it may become part of the requirement for consistency.\
This means that the results have to remain the same when modifications are made.

### propose an IMPROVEment

You don't need to fill up an IMPROVE request from the "Issues board" to propose an IMPROVE request.\
These are just some API and semantic request and should be simple to read (an understand) and pass consistency test.

## Accepted changes

When a change is proposed, the AUTHORS file will be modified.\
For example, if someone proposed a fix for the BUG number 218, the following line will be added:
```
<git user name> FIX the BUG number 218 on YYYY/MM/DD.
```
Some additional comment may be added, mostly to praise an exceptional contribution.\
If you want a specific name written in the AUTHORS file, please state so in the pull request.\
If you want your contribution to remain anonymous, please also say so in the PR.\
You can also directly add the line to the AUTHORS file, but it is better to let us handle it.







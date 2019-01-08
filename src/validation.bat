@set TestExe=..\evaltool\test.exe
@set GroundTruth=..\evaltool\validation-set.data
@set Ans=..\evaltool\ans.txt
@set Output=..\evaltool\result.txt

call %TestExe% %GroundTruth% %Ans% %Output%

pause
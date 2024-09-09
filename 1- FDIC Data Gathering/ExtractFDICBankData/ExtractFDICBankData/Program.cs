using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using System.Reflection.Emit;
/* Retreives YAML specification of financial reporting api from FDIC and 
 * downloads all numeric properties along with BKCLASS, CERT, and REPDTE */


namespace ExtractFDICBankData
{
    using System;
    using System.IO;
    using System.Net;
    using System.Text.RegularExpressions;
    using YamlDotNet.RepresentationModel;

    class Program
    {
        static void Main()
        {
            List<string> InstitutionProperties = new List<string>();
            List<string> InstitutionDesc = new List<string>();
            if (!File.Exists("risview_properties.yaml"))
                using (var client = new WebClient())
                {
                    //Download financial API YAML definitions. https://banks.data.fdic.gov/docs/
                    client.DownloadFile("https://banks.data.fdic.gov/docs/risview_properties.yaml", "risview_properties.yaml");
                }
            var inputYaml = File.ReadAllText(@"risview_properties.yaml");

            var yamlStream = new YamlStream();
            yamlStream.Load(new StringReader(inputYaml));

            // Navigate to the properties.data.properties node
            var rootNode = (YamlMappingNode)yamlStream.Documents[0].RootNode;
            var propertiesNode = (YamlMappingNode)rootNode.Children[new YamlScalarNode("properties")];
            var dataNode = (YamlMappingNode)propertiesNode.Children[new YamlScalarNode("data")];
            var propertiesDataNode = (YamlMappingNode)dataNode.Children[new YamlScalarNode("properties")];

            // Extract and print property names
            foreach (var propertyNode in propertiesDataNode.Children)
            {
                var propertyList = propertyNode.Value.AllNodes.ToList();
                
                bool isNum = false;
                bool isRatio = false;

                int index = 0;
                string tempDesc = "";
                foreach (var property in propertyList)
                {
                    if(property.ToString() == "number")
                        isNum = true;
                    else if(property.ToString() == "title")
                    {
                        tempDesc = propertyList[index+1].ToString();
                    } 
                    index++;
                }
                string type = propertyNode.Value.AllNodes.ToList()[2].ToString();
                var propertyName = ((YamlScalarNode)propertyNode.Key).Value;
                if (isNum && propertyName!="CERT")
                {
                    InstitutionDesc.Add(tempDesc);
                    InstitutionProperties.Add(propertyName);
                    
                }
                tempDesc = "";
            }

            //OUTPUTS Long Text Description and short-text property name used by datasets
            File.WriteAllLines("InstitutionDesc.txt", InstitutionDesc);
            File.WriteAllLines("InstitutionProperties.txt", InstitutionProperties);

            //Begin download request from current available specified numeric values

            int capped = 0; //Starting point for captured institution properties, can be adjusted based on if program fails midway through pulls.
            List<String> tempFields;
            while (capped < InstitutionProperties.Count)
            {
                if(InstitutionProperties.Count - capped < 100)
                    tempFields = InstitutionProperties.Skip(capped).Take(InstitutionProperties.Count - capped).ToList();
                else
                    tempFields = InstitutionProperties.Skip(capped).Take(100).ToList();
                tempFields.Add("CERT");
                tempFields.Add("REPDTE");
                tempFields.Add("BKCLASS");
                capped += 100;
                string tempFieldStr = string.Join("%2C", tempFields);
                string apiUrl = $"https://banks.data.fdic.gov/api/financials?filters=REPDTE%3A%5B%222005-12-31%22%20TO%20%222025-12-31%22%5D&fields={tempFieldStr}&sort_by=REPDTE&sort_order=DESC&limit=10000&format=csv&download=true&filename=2022-2023-Quarterly";

           
                string outputFile = $"TotalResults{Math.Round((double)(capped/100)).ToString()}.csv";
                int offset = 0;
                int lastOffset = -1;
                using (WebClient client = new WebClient())
                {


                    string url = $"{apiUrl}&offset={offset}";

                    // Download the file
                    client.DownloadFile(url, outputFile);

                    while (lastOffset < offset)
                    {
                        //Calculate the number of rows returned in the previous query

                        //offset = rowCount - 1;
                        //Increment the offset


                        //Update the URL with the new offset
                        url = $"{apiUrl}&offset={offset}";

                        // Download the next file
                        string tempFile = $"temp_{offset}.csv";
                        client.DownloadFile(url, tempFile);
                        int rowCount = File.ReadAllLines(tempFile).Length - 1;
                        if (rowCount <= 0) { break; }
                        lastOffset = offset;
                        offset += rowCount;

                        // Concatenate the downloaded file to the output file
                        string[] lines = File.ReadAllLines(tempFile);
                        File.AppendAllText(outputFile, Environment.NewLine + string.Join(Environment.NewLine, lines.Skip(1).ToList()));
                        //File.AppendAllText(outputFile, File.ReadAllText(tempFile));
                        // Delete the temporary file
                        File.Delete(tempFile);
                    }

                }
            }


        }
    }

}
